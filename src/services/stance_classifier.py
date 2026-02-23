"""
Stance Classifier using NLI (Natural Language Inference).

Classifies legal articles and precedents as:
- SUPPORT: supports the claim (for Reasoner)
- AGAINST: challenges/contradicts the claim (for CounterReasoner)
- NEUTRAL: neither clearly supports nor challenges

Uses LLM-based NLI for stance detection.
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.tools.prompt_registry import render_prompt  # noqa: E402
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402


class Stance(Enum):
    """Stance classification result."""

    SUPPORT = "support"
    AGAINST = "against"
    NEUTRAL = "neutral"


@dataclass
class StanceResult:
    """Result of stance classification for a single item."""

    item: dict
    stance: Stance
    confidence: str  # "high", "medium", "low"
    reasoning: str = ""


# Mapping from qualitative confidence labels to numeric values
# used by the AQA δ-scoring formula.
CONFIDENCE_TO_FLOAT: dict[str, float] = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.4,
}


class StanceClassifier:
    """
    Classifies articles and precedents as supporting or opposing a legal claim.

    Uses NLI-style prompting to determine:
    - SUPPORT: The article/precedent can be used to support the claim
    - AGAINST: The article/precedent can be used to challenge the claim
    - NEUTRAL: No clear stance or equally applicable to both sides
    """

    def __init__(self, llm: Optional[ChatGroq] = None):
        """Initialize the stance classifier."""
        self._llm = llm

    @property
    def llm(self) -> ChatGroq:
        """Lazy initialization of LLM with resilient key management."""
        if self._llm is None:
            self._llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.classifier_max_tokens,
            )
        return self._llm

    def classify_statute(self, claim: str, statute: dict) -> StanceResult:
        """
        Classify a single statute as supporting or opposing the claim.

        Args:
            claim: The legal claim
            statute: Dict with 'articolo', 'titolo', 'testo', 'source'

        Returns:
            StanceResult with stance and confidence
        """
        article_num = statute.get("articolo", "N/A")
        title = statute.get("titolo", "")
        text = statute.get("testo", "")[: settings.truncation_statute_text]
        source_key = statute.get("source")
        if source_key == "codice_civile":
            source = "c.c."
        elif source_key == "codice_penale":
            source = "c.p."
        elif source_key == "codice_amministrativo":
            source = "L. 241/1990"
        else:
            source = str(source_key or "codice")

        try:
            support_answer = self._ask_statute_axis(
                claim=claim,
                article_num=article_num,
                source=source,
                title=title,
                text=text,
                axis="SUPPORT",
            )
            against_answer = self._ask_statute_axis(
                claim=claim,
                article_num=article_num,
                source=source,
                title=title,
                text=text,
                axis="AGAINST",
            )
            return self._combine_axis_votes(
                item=statute,
                support_vote=self._parse_yes_no(support_answer),
                against_vote=self._parse_yes_no(against_answer),
                source_label=f"Art. {article_num}",
                raw_support=support_answer,
                raw_against=against_answer,
            )
        except Exception as e:
            print(f"⚠️ Stance classification failed for Art. {article_num}: {e}")
            return StanceResult(
                item=statute,
                stance=Stance.NEUTRAL,
                confidence="low",
                reasoning=f"Classification error: {e}",
            )

    def classify_precedent(self, claim: str, precedent: dict) -> StanceResult:
        """
        Classify a single precedent as supporting or opposing the claim.

        Args:
            claim: The legal claim
            precedent: Dict with 'title', 'summary', etc.

        Returns:
            StanceResult with stance and confidence
        """
        title = precedent.get("title", "Untitled")
        summary = precedent.get("summary", "")[: settings.truncation_summary]

        try:
            support_answer = self._ask_precedent_axis(
                claim=claim,
                title=title,
                summary=summary,
                axis="SUPPORT",
            )
            against_answer = self._ask_precedent_axis(
                claim=claim,
                title=title,
                summary=summary,
                axis="AGAINST",
            )
            return self._combine_axis_votes(
                item=precedent,
                support_vote=self._parse_yes_no(support_answer),
                against_vote=self._parse_yes_no(against_answer),
                source_label=f"precedent '{title[:50]}'",
                raw_support=support_answer,
                raw_against=against_answer,
            )
        except Exception as e:
            print(f"⚠️ Stance classification failed for precedent '{title[:50]}': {e}")
            return StanceResult(
                item=precedent,
                stance=Stance.NEUTRAL,
                confidence="low",
                reasoning=f"Classification error: {e}",
            )

    def classify_statutes_batch(
        self, claim: str, statutes: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Classify multiple statutes and separate into support vs against.

        Args:
            claim: The legal claim
            statutes: List of statute dicts

        Returns:
            Tuple of (supporting_statutes, opposing_statutes)
        """
        supporting = []
        opposing = []
        neutral = []

        print(f"🔍 [StanceClassifier] Classifying {len(statutes)} statutes...")

        for statute in statutes:
            result = self.classify_statute(claim, statute)
            article = statute.get("articolo", "N/A")
            statute["_stance_label"] = result.stance.value
            statute["_stance_confidence"] = CONFIDENCE_TO_FLOAT.get(
                result.confidence.lower(), 0.5
            )

            if result.stance == Stance.SUPPORT:
                supporting.append(statute)
                print(f"  ✅ Art. {article}: SUPPORTO")
            elif result.stance == Stance.AGAINST:
                opposing.append(statute)
                print(f"  ❌ Art. {article}: CONTRO")
            else:
                neutral.append(statute)
                print(f"  ⚖️ Art. {article}: NEUTRALE")

        print(
            f"📊 [StanceClassifier] Result: {len(supporting)} support, {len(opposing)} against, {len(neutral)} neutral"
        )
        return supporting, opposing, neutral

    def classify_precedents_batch(
        self, claim: str, precedents: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Classify multiple precedents and separate into support vs against.

        Args:
            claim: The legal claim
            precedents: List of precedent dicts

        Returns:
            Tuple of (supporting_precedents, opposing_precedents)
        """
        supporting = []
        opposing = []
        neutral = []

        print(f"🔍 [StanceClassifier] Classifying {len(precedents)} precedents...")

        for precedent in precedents:
            result = self.classify_precedent(claim, precedent)
            title = precedent.get("title", "Untitled")[:50]

            # Annotate the precedent dict so downstream components
            # (ASPIC formatter, AQA engine) can use the stance info.
            precedent["_stance_label"] = result.stance.value
            precedent["_stance_confidence"] = CONFIDENCE_TO_FLOAT.get(
                result.confidence.lower(), 0.5
            )

            if result.stance == Stance.SUPPORT:
                supporting.append(precedent)
                print(f"  ✅ '{title}...': SUPPORTO")
            elif result.stance == Stance.AGAINST:
                opposing.append(precedent)
                print(f"  ❌ '{title}...': CONTRO")
            else:
                neutral.append(precedent)
                print(f"  ⚖️ '{title}...': NEUTRALE")

        print(
            f"📊 [StanceClassifier] Result: {len(supporting)} support, {len(opposing)} against, {len(neutral)} neutral"
        )
        return supporting, opposing, neutral

    def _build_statute_prompt(
        self,
        claim: str,
        article_num: str,
        source: str,
        title: str,
        text: str,
        stance_axis: str,
    ) -> str:
        """Build binary NLI prompt for statute classification on one axis."""
        return render_prompt(
            "stance_classifier.statute",
            claim=claim,
            article_num=article_num,
            source=source,
            title=title,
            text=text,
            stance_axis=stance_axis,
        )

    def _build_precedent_prompt(
        self, claim: str, title: str, summary: str, stance_axis: str
    ) -> str:
        """Build binary NLI prompt for precedent classification on one axis."""
        return render_prompt(
            "stance_classifier.precedent",
            claim=claim,
            title=title,
            summary=summary,
            stance_axis=stance_axis,
        )

    def _ask_statute_axis(
        self,
        claim: str,
        article_num: str,
        source: str,
        title: str,
        text: str,
        axis: str,
    ) -> str:
        prompt = self._build_statute_prompt(
            claim=claim,
            article_num=article_num,
            source=source,
            title=title,
            text=text,
            stance_axis=axis,
        )
        response = resilient_chat_call(self.llm, [HumanMessage(content=prompt)])
        return (response.content or "").strip()

    def _ask_precedent_axis(
        self, claim: str, title: str, summary: str, axis: str
    ) -> str:
        prompt = self._build_precedent_prompt(
            claim=claim,
            title=title,
            summary=summary,
            stance_axis=axis,
        )
        response = resilient_chat_call(self.llm, [HumanMessage(content=prompt)])
        return (response.content or "").strip()

    @staticmethod
    def _parse_yes_no(answer: str) -> bool | None:
        token = (answer or "").strip().upper().split()
        head = token[0] if token else ""
        if head in {"YES", "SI", "SÌ"}:
            return True
        if head in {"NO"}:
            return False

        upper = (answer or "").upper()
        if "YES" in upper:
            return True
        if " NO" in f" {upper}":
            return False
        return None

    def _combine_axis_votes(
        self,
        item: dict,
        support_vote: bool | None,
        against_vote: bool | None,
        source_label: str,
        raw_support: str,
        raw_against: str,
    ) -> StanceResult:
        """
        Map dual binary votes into a neutral, symmetric 3-way stance.

        Mapping:
        - YES/NO  -> SUPPORT
        - NO/YES  -> AGAINST
        - NO/NO   -> NEUTRAL
        - YES/YES -> NEUTRAL (conflict)
        - any None -> NEUTRAL (unparseable)
        """
        if support_vote is None or against_vote is None:
            return StanceResult(
                item=item,
                stance=Stance.NEUTRAL,
                confidence="low",
                reasoning=(
                    f"Unparseable binary stance output for {source_label}: "
                    f"SUPPORT='{raw_support}' | AGAINST='{raw_against}'"
                ),
            )

        if support_vote and not against_vote:
            return StanceResult(item=item, stance=Stance.SUPPORT, confidence="high")
        if against_vote and not support_vote:
            return StanceResult(item=item, stance=Stance.AGAINST, confidence="high")
        if support_vote and against_vote:
            return StanceResult(
                item=item,
                stance=Stance.NEUTRAL,
                confidence="low",
                reasoning=(
                    f"Conflicting binary stance for {source_label}: both SUPPORT and AGAINST are YES"
                ),
            )
        return StanceResult(item=item, stance=Stance.NEUTRAL, confidence="medium")
