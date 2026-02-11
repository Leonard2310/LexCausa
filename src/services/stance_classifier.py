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
        source = "c.c." if statute.get("source") == "codice_civile" else "c.p."

        prompt = self._build_statute_prompt(claim, article_num, source, title, text)

        try:
            response = resilient_chat_call(self.llm, [HumanMessage(content=prompt)])
            answer = response.content.strip().upper()
            return self._parse_response(statute, answer)
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

        prompt = self._build_precedent_prompt(claim, title, summary)

        try:
            response = resilient_chat_call(self.llm, [HumanMessage(content=prompt)])
            answer = response.content.strip().upper()
            return self._parse_response(precedent, answer)
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
        self, claim: str, article_num: str, source: str, title: str, text: str
    ) -> str:
        """Build NLI prompt for statute classification."""
        return f"""Task: Classify whether this legal article SUPPORTS or OPPOSES the claim.

CLAIM (Legal thesis):
"{claim}"

ARTICLE (Art. {article_num} {source} - {title}):
"{text}"

CLASSIFICATION RULES:
- SUPPORT: The article provides legal basis that REINFORCES or VALIDATES the claim, or sets formal requirements whose violation strengthens the claim.
- AGAINST: The article provides grounds to CHALLENGE, LIMIT, or CONTRADICT the claim.
- NEUTRAL: The article is equally applicable to both positions or irrelevant.
- DEFAULT TO SUPPORT unless the article clearly limits/contradicts the claim.

Consider:
- Does the article establish rights that support the claimant?
- Does the article impose limits, exceptions, or defenses against the claim?
- Does the article define conditions that may not be met?

Respond with EXACTLY one word: SUPPORT, AGAINST, or NEUTRAL
No punctuation. No explanations."""

    def _build_precedent_prompt(self, claim: str, title: str, summary: str) -> str:
        """Build NLI prompt for precedent classification."""
        return f"""Task: Classify whether this judicial precedent SUPPORTS or OPPOSES the claim.

CLAIM (Legal thesis):
"{claim}"

PRECEDENT ({title}):
"{summary}"

CLASSIFICATION RULES:
- SUPPORT: The precedent establishes principles that REINFORCE the claim
- AGAINST: The precedent establishes principles that CHALLENGE or LIMIT the claim
- NEUTRAL: The precedent is not clearly applicable to either position

Respond with EXACTLY one word: SUPPORT, AGAINST, or NEUTRAL
No punctuation. No explanations."""

    def _parse_response(self, item: dict, answer: str) -> StanceResult:
        """Parse LLM response into StanceResult."""
        # Extract first word
        token = answer.split()[0] if answer else ""

        if "SUPPORT" in token or "SUPPORTO" in token:
            return StanceResult(item=item, stance=Stance.SUPPORT, confidence="high")
        elif "AGAINST" in token or "CONTRO" in token:
            return StanceResult(item=item, stance=Stance.AGAINST, confidence="high")
        elif "NEUTRAL" in token or "NEUTRALE" in token:
            return StanceResult(item=item, stance=Stance.NEUTRAL, confidence="medium")
        else:
            # Fallback: check for keywords in full response
            if "SUPPORT" in answer or "SUPPORTO" in answer:
                return StanceResult(
                    item=item, stance=Stance.SUPPORT, confidence="medium"
                )
            elif "AGAINST" in answer or "CONTRO" in answer:
                return StanceResult(
                    item=item, stance=Stance.AGAINST, confidence="medium"
                )
            else:
                return StanceResult(
                    item=item,
                    stance=Stance.NEUTRAL,
                    confidence="low",
                    reasoning=f"Unparseable response: {answer}",
                )
