"""
Base Agent class for LexCausa reasoning system.

Provides common functionality for all agents including:
- LLM initialization (Groq)
- Neo4j connection
- Logging and error handling
- Common extraction methods for tool messages
"""

import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .tools.prompt_registry import render_prompt

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import (  # noqa: E402
    get_chat_groq,
    resilient_chat_call,
    resilient_chat_stream,
)


@dataclass
class AgentConfig:
    """Configuration for LexCausa agents."""

    # Groq LLM settings
    groq_api_key: str = field(default_factory=lambda: settings.groq_api_key)
    model_name: str = field(default_factory=lambda: settings.groq_models[0])
    temperature: float = field(default_factory=lambda: settings.llm_temperature)
    max_tokens: int = field(default_factory=lambda: settings.llm_max_tokens)

    # Neo4j settings
    neo4j_uri: str = field(default_factory=lambda: settings.neo4j_uri)
    neo4j_user: str = field(default_factory=lambda: settings.neo4j_user)
    neo4j_password: str = field(default_factory=lambda: settings.neo4j_password)

    def __post_init__(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    claim: str
    causality_type: str  # Materiale, Giuridica, Concause
    causality_details: dict
    arguments: list[dict]
    supporting_statutes: list[dict]
    supporting_precedents: list[dict]
    reasoning_chain: list[str]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causality_type": self.causality_type,
            "causality_details": self.causality_details,
            "arguments": self.arguments,
            "supporting_statutes": self.supporting_statutes,
            "supporting_precedents": self.supporting_precedents,
            "reasoning_chain": self.reasoning_chain,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for LexCausa agents.

    All agents (Reasoner, CounterReasoner, PolisherEvaluator) inherit from this class.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self._llm: Optional[ChatGroq] = None
        self._fact_lock_check_cache: dict[tuple[str, str], tuple[bool, str]] = {}
        self._nli_relation_cache: dict[tuple[str, str], str] = {}

    @property
    def llm(self) -> ChatGroq:
        """Lazy initialization of LLM with resilient key management."""
        if self._llm is None:
            self._llm = get_chat_groq(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.groq_api_key or None,
            )
        return self._llm

    def _resilient_llm_invoke(self, messages, **kwargs):
        """Invoke LLM with automatic retry, key rotation, and model fallback."""
        stream_callback = kwargs.pop("stream_callback", None)
        model_order = self._resilient_model_order()
        if stream_callback is not None:
            return resilient_chat_stream(
                self.llm,
                messages,
                on_token=stream_callback,
                model_order=model_order,
                **kwargs,
            )
        return resilient_chat_call(
            self.llm,
            messages,
            model_order=model_order,
            **kwargs,
        )

    def _resilient_retrieval_llm_invoke(self, messages, **kwargs):
        """Invoke LLM for retrieval-side filtering with dedicated deterministic temperature."""
        stream_callback = kwargs.pop("stream_callback", None)
        model_order = self._resilient_model_order()
        retrieval_temp = 0.0
        max_tokens = getattr(self.config, "max_tokens", settings.llm_max_tokens)

        def _llm_factory(api_key: str, model: str):
            return get_chat_groq(
                model=model,
                temperature=retrieval_temp,
                max_tokens=max_tokens,
                api_key=api_key or None,
            )

        if stream_callback is not None:
            return resilient_chat_stream(
                _llm_factory,
                messages,
                on_token=stream_callback,
                model_order=model_order,
                **kwargs,
            )
        return resilient_chat_call(
            _llm_factory,
            messages,
            model_order=model_order,
            **kwargs,
        )

    def _resilient_model_order(self) -> list[str] | None:
        """Optional per-agent model fallback order override for resilient calls."""
        return None

    def close(self):
        """Close resources held by the agent."""
        pass

    @abstractmethod
    def run(self, claim: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the agent's main reasoning task.

        Args:
            claim: The legal claim to reason about.
            *args: Positional arguments specific to each agent.
            **kwargs: Keyword arguments specific to each agent.

        Returns:
            Agent-specific result.
        """
        pass

    def _log(self, message: str, level: str = "info"):
        """Simple logging helper."""
        emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(
            level, "•"
        )
        print(f"{emoji} [{self.__class__.__name__}] {message}")

    # ------------------------------------------------------------------
    # Step repetition detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_garbage_text(text: str, min_words: int = 15) -> bool:
        """Return ``True`` if *text* looks like degenerate LLM output.

        Detects token-repetition loops (e.g. the model outputs the same
        word hundreds of times) by checking:
        1. **Unique-word ratio**: if unique_words / total_words < 0.15
           the text is almost certainly a repetition loop.
        2. **Dominant-token ratio**: if any single token accounts for
           more than 40 % of all tokens, it's degenerate.
        3. **Non-Latin dominance**: if more than 50 % of alphabetical
           characters are outside the Latin/extended-Latin range
           (e.g. CJK, Cyrillic) the model switched language randomly.

        Short texts (< *min_words* words) are never flagged.
        """
        words = text.split()
        n_words = len(words)
        if n_words < min_words:
            return False

        # 1. Unique-word ratio
        unique = set(w.lower() for w in words)
        if len(unique) / n_words < 0.15:
            return True

        # 2. Dominant-token ratio
        from collections import Counter

        counts = Counter(w.lower() for w in words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / n_words > 0.40:
            return True

        # 3. Non-Latin dominance
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            non_latin = sum(
                1 for c in alpha_chars if ord(c) > 0x024F  # beyond Latin Extended-B
            )
            if non_latin / len(alpha_chars) > 0.50:
                return True

        return False

    @staticmethod
    def _is_repetitive_step(
        new_step: str,
        existing_steps: list[str],
        threshold: float = 0.70,
    ) -> bool:
        """Return ``True`` if *new_step* is too similar to any existing step.

        Uses **word-level Jaccard similarity**: if the fraction of shared
        words between the candidate and any earlier step exceeds
        *threshold* the step is considered a repetition.

        A threshold of 0.70 catches near-identical paraphrases while
        allowing steps that share some legal vocabulary but discuss
        genuinely different aspects.
        """
        new_words = set(new_step.lower().split())
        if not new_words:
            return False
        for prev in existing_steps:
            prev_words = set(prev.lower().split())
            if not prev_words:
                continue
            intersection = new_words & prev_words
            union = new_words | prev_words
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= threshold:
                return True
        return False

    @staticmethod
    def _split_reasoning_clauses(step_text: str) -> list[str]:
        """Split a step into comparable clauses for semantic self-consistency checks."""
        text = re.sub(r"\s+", " ", (step_text or "").strip())
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?;])\s+", text)
        clauses: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip(" -")
            if not sentence:
                continue
            parts = re.split(
                r"\b(?:ma|tuttavia|per[oò]|invece|al contrario|nondimeno)\b",
                sentence,
                flags=re.IGNORECASE,
            )
            for part in parts:
                chunk = part.strip(" ,.-")
                if len(chunk) >= 20:
                    clauses.append(chunk)
        return clauses

    def _is_step_self_consistent(self, step_text: str) -> bool:
        """Reject internal contradictions using semantic (NLI) checks across clauses."""
        text = re.sub(r"\s+", " ", (step_text or "").strip())
        if not text:
            return False

        clauses = self._split_reasoning_clauses(text)
        if len(clauses) < 2:
            return True

        clauses = clauses[:5]  # keep latency bounded
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                a = clauses[i]
                b = clauses[j]
                rel_ab = self._nli_relation(
                    target_text=a,
                    attacker_text=b,
                    actor_label="BaseAgent",
                )
                if rel_ab != "contradiction":
                    continue
                rel_ba = self._nli_relation(
                    target_text=b,
                    attacker_text=a,
                    actor_label="BaseAgent",
                )
                if rel_ba == "contradiction":
                    return False
        return True

    def _is_step_fact_consistent_with_claim(
        self,
        *,
        claim: str,
        candidate_step: str,
        actor_label: str = "Agent",
    ) -> tuple[bool, str]:
        """Reject steps that contradict explicit facts stated in the claim.

        Shared helper used by Reasoner and CounterReasoner.
        """
        claim_text = (claim or "").strip()
        step_text = (candidate_step or "").strip()
        if not claim_text or not step_text:
            return True, ""

        cache_key = (claim_text, step_text)
        cached = self._fact_lock_check_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "base.fact_lock_check",
            claim=claim_text,
            candidate_step=step_text,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip().upper()
            if "CONTRADICT" in answer:
                result = (False, "contradicts explicit claim fact")
            else:
                result = (True, "")
        except Exception as exc:
            # Do not fail closed on checker outages/rate limits.
            self._log(
                f"⚠️ {actor_label} fact-lock check failed (fallback keep): {exc}",
                "warning",
            )
            result = (True, "")

        self._fact_lock_check_cache[cache_key] = result
        return result

    def _nli_relation(
        self,
        *,
        target_text: str,
        attacker_text: str,
        actor_label: str = "Agent",
    ) -> str:
        """
        Classify semantic relation between two passages using the shared legal NLI prompt.

        Returns one of: ``contradiction``, ``entailment``, ``neutral``.
        """
        target = re.sub(r"\s+", " ", (target_text or "").strip())[
            : settings.truncation_nli_text
        ]
        attacker = re.sub(r"\s+", " ", (attacker_text or "").strip())[
            : settings.truncation_nli_text
        ]
        if not target or not attacker:
            return "neutral"

        cache_key = (target, attacker)
        cached = self._nli_relation_cache.get(cache_key)
        if cached is not None:
            return cached

        system_prompt = render_prompt("nlp_utils.nli_system")
        user_prompt = render_prompt(
            "nlp_utils.nli_user",
            target_text=target,
            attacker_text=attacker,
        )

        try:
            resp = self._resilient_llm_invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            answer = (resp.content or "").strip().upper()
            if "CONTRADICTION" in answer:
                label = "contradiction"
            elif "ENTAILMENT" in answer:
                label = "entailment"
            else:
                label = "neutral"
        except Exception as exc:
            self._log(
                f"⚠️ {actor_label} NLI relation check failed (fallback neutral): {exc}",
                "warning",
            )
            label = "neutral"

        self._nli_relation_cache[cache_key] = label
        return label

    def _extract_reasoning_chain(self, response: str) -> list[str]:
        """Extract reasoning chain from response with improved pattern matching."""
        chain = []
        lines = response.split("\n")

        # Patterns that indicate start of reasoning chain section
        chain_markers = [
            "catena",
            "ragionamento",
            "chain",
            "reasoning",
            "conclusione",
            "sintesi",
            "riepilogo",
            "summary",
            "passaggi",
            "steps",
            "argomentazione",
        ]

        # Pattern to detect pure norm-reference lines like:
        # "**Offesa ingiusta**: Art. 52 c.p."  or  "Causalità: Art. 40 c.p. e Art. 41 c.p."
        # These are short summary items, NOT reasoning steps.
        _norm_ref_pattern = re.compile(
            r"^(?:\*\*)?[^:]{2,40}(?:\*\*)?:\s*Art\.?\s*\d",
            re.IGNORECASE,
        )

        in_chain = False
        for line in lines:
            line = line.strip()
            lower_line = line.lower()

            # Check if this line starts a chain section
            if any(marker in lower_line for marker in chain_markers):
                in_chain = True
                # If the marker line itself has content after ":", extract it
                if ":" in line:
                    content = line.split(":", 1)[1].strip()
                    if content and len(content) > 10:
                        chain.append(content)
                continue

            # Detect a new bold section header that is NOT a chain section
            # (e.g. **Premessa**, **Norma**, **Nesso Causale**).
            # When we hit one of these while already in_chain, stop collecting.
            if in_chain and line.startswith("**") and ":" in line:
                header_text = line.split(":", 1)[0].replace("*", "").strip().lower()
                if not any(marker in header_text for marker in chain_markers):
                    in_chain = False
                    continue

            if in_chain and line:
                # Skip markdown bold section headers (e.g. **Ulteriore Norma**:)
                if line.startswith("**"):
                    continue
                # Numbered items (1., 2., etc.)
                if line[0].isdigit() and len(line) > 2:
                    step_text = line.lstrip("0123456789.) ")
                    # Skip pure norm-reference items
                    # e.g. "**Offesa ingiusta**: Art. 52 c.p."
                    if _norm_ref_pattern.match(step_text):
                        continue
                    chain.append(step_text)
                # Bullet points
                elif line.startswith(("-", "•", "—", "→")):
                    chain.append(line.lstrip("-•—→ "))
                # Arrow notation
                elif "→" in line or "->" in line:
                    chain.append(line)
                # Single * bullet (not markdown bold **)
                elif line.startswith("*") and not line.startswith("**"):
                    chain.append(line.lstrip("* "))

        return chain if chain else ["Catena di ragionamento non disponibile."]

    def _sanitize_reasoning_chain(
        self, chain: list[str], precedents: list[dict]
    ) -> list[str]:
        """Clean and sanitize reasoning chain, handling precedent mentions."""

        def _is_citation_removed(text: str) -> bool:
            t = text.strip().lower()
            return (
                t.startswith("[citation removed")
                or t.startswith("citation removed")
                or "[citation removed" in t
            )

        if precedents:
            return [
                self._clean_chain_step(step)
                for step in chain
                if not _is_citation_removed(step)
            ]

        sanitized = []
        for step in chain:
            cleaned = self._clean_chain_step(step)
            if _is_citation_removed(cleaned):
                continue
            lower = cleaned.lower()
            mentions_precedent = "precedent" in lower or "precedente" in lower
            mentions_absence = "nessun" in lower or "nessuna" in lower
            if mentions_precedent and not mentions_absence:
                continue
            sanitized.append(cleaned)

        if not any(
            "precedent" in s.lower() or "precedente" in s.lower() for s in sanitized
        ):
            sanitized.append("Precedents: none found.")

        return sanitized

    def _clean_chain_step(self, step: str) -> str:
        """Clean a single chain step."""
        cleaned = step.strip()
        if "**" in cleaned:
            cleaned = cleaned.replace("**", "")
        return cleaned.strip()

    def _has_valid_reasoning_chain(self, aspic_ir: dict) -> bool:
        """Check if ASPIC_IR contains valid reasoning chain nodes (S1, S2, …).

        A valid reasoning chain must have at least one step whose id
        starts with 'S' (e.g. S1, S2, S3).  Used to decide whether
        the LLM generation should be retried.
        """
        if not aspic_ir:
            return False
        chain = aspic_ir.get("reasoning_chain", [])
        if not chain:
            return False
        return any(
            isinstance(step, dict) and step.get("id", "").startswith("S")
            for step in chain
        )

    # ------------------------------------------------------------------
    # Pre-retrieval filters (shared across agents)
    # ------------------------------------------------------------------

    def filter_irrelevant_statutes(
        self, claim: str, statutes: list[dict]
    ) -> list[dict]:
        """Filter statutes using LLM one by one.

        Only discard when clearly unrelated; default to keeping on ambiguity.
        This is a PUBLIC method called from api_server for pre-filtering.
        """
        if not statutes:
            self._log("No statutes to filter", "info")
            return statutes

        self._log(f"🔍 Filtering relevance: {len(statutes)} statutes initially")

        relevant_statutes = []
        for idx, statute in enumerate(statutes, start=1):
            article_number = statute.get("articolo", "N/A")
            article_title = statute.get("titolo", "Untitled")
            article_desc = statute.get("testo", "Untitled")

            prompt = render_prompt(
                "base.filter_relevant_statutes",
                claim=claim,
                article_number=article_number,
                article_title=article_title,
                article_desc=article_desc,
            )

            try:
                response = self._resilient_retrieval_llm_invoke(
                    [HumanMessage(content=prompt)]
                )
                answer = response.content.strip().upper()
            except Exception as e:
                self._log(
                    f"⚠️ LLM call failed for article {article_number}: {e}", "warning"
                )
                answer = "YES"

            token = answer.split()[0] if answer else ""
            keep = token != "NO" and (
                token == "YES" or "YES" in answer or "NO" not in answer
            )

            if keep:
                relevant_statutes.append(statute)
                self._log(
                    f"✅ Keeping article [{idx}] {article_number} - {article_title}"
                )
            else:
                self._log(
                    f"❌ Discarding article [{idx}] {article_number} - {article_title}",
                    "warning",
                )

        self._log(f"📊 Result: {len(relevant_statutes)}/{len(statutes)} statutes kept")
        return relevant_statutes

    def _extract_legal_context(self, claim: str) -> str:
        """Extract a short legal context label used by applicability filtering.

        Returns a compact line containing domain/party relationship/procedural
        posture so statute applicability can be evaluated more strictly.
        """
        prompt = render_prompt("base.extract_legal_context", claim=claim)

        try:
            response = self._resilient_retrieval_llm_invoke(
                [HumanMessage(content=prompt)]
            )
            context = (response.content or "").strip().replace("\n", " ")
            context = re.sub(r"\s+", " ", context)
            if context:
                self._log(f"🧭 Legal context extracted: {context[:120]}")
                return context[:200]
        except Exception as e:
            self._log(f"⚠️ Legal context extraction failed: {e}", "warning")

        fallback = "General legal dispute context"
        self._log(f"🧭 Legal context fallback: {fallback}")
        return fallback

    def filter_applicable_statutes(
        self, claim: str, statutes: list[dict], legal_context: str
    ) -> list[dict]:
        """Second-stage filter: keep statutes that are legally applicable.

        This is called after topical relevance filtering and removes statutes
        that are related by keywords but inapplicable to the case posture.
        """
        if not statutes:
            return statutes

        self._log(f"🎯 Checking applicability: {len(statutes)} statutes")
        applicable_statutes: list[dict] = []
        for idx, statute in enumerate(statutes, start=1):
            article_number = statute.get("articolo", "N/A")
            article_title = statute.get("titolo", "Untitled")
            article_text = statute.get("testo", "") or ""

            prompt = render_prompt(
                "base.filter_applicable_statutes",
                claim=claim,
                legal_context=legal_context,
                article_number=article_number,
                article_title=article_title,
                article_text=article_text[:500],
            )

            try:
                response = self._resilient_retrieval_llm_invoke(
                    [HumanMessage(content=prompt)]
                )
                answer = (response.content or "").strip().upper()
            except Exception as e:
                self._log(
                    f"⚠️ Applicability check failed for {article_number}: {e}",
                    "warning",
                )
                answer = "YES"  # safe default: keep on error

            token = answer.split()[0] if answer else ""
            keep = token != "NO" and (
                token == "YES" or "YES" in answer or "NO" not in answer
            )

            if keep:
                applicable_statutes.append(statute)
                self._log(f"✅ APPLICABLE [{idx}] {article_number} - {article_title}")
            else:
                self._log(
                    f"❌ NOT APPLICABLE [{idx}] {article_number} - {article_title}",
                    "warning",
                )

        self._log(
            f"📊 Applicability result: {len(applicable_statutes)}/{len(statutes)} kept"
        )
        return applicable_statutes

    def filter_irrelevant_precedents(
        self, claim: str, precedents: list[dict]
    ) -> list[dict]:
        """Filter precedents by substantive legal relevance to the claim.

        Uses a single LLM call per precedent that evaluates whether the
        legal question, ratio decidendi, or established principle of the
        precedent is concretely applicable to the claim — regardless of
        whether they share the same legal domain.

        On LLM error the precedent is KEPT (same safe-default as statutes).
        """
        if not precedents:
            return precedents

        self._log(f"🔍 Filtering {len(precedents)} precedents (relevance mode)")

        relevant: list[dict] = []
        for idx, precedent in enumerate(precedents, start=1):
            title = precedent.get("title", "Untitled")
            summary = precedent.get("summary", "")
            materia = precedent.get("materia", "")

            materia_line = f'\nDomain: "{materia}"' if materia else ""

            prompt = render_prompt(
                "base.filter_precedents",
                claim=claim,
                title=title,
                materia_line=materia_line,
                summary=summary[:600],
            )

            try:
                response = self._resilient_retrieval_llm_invoke(
                    [HumanMessage(content=prompt)]
                )
                answer = response.content.strip().upper()
            except Exception as e:
                self._log(
                    f"⚠️ LLM call failed for precedent [{idx}] {title}: {e}",
                    "warning",
                )
                answer = "YES"  # safe default: keep on error

            token = answer.split()[0] if answer else ""
            keep = token != "NO" and (
                token == "YES" or "YES" in answer or "NO" not in answer
            )

            if keep:
                relevant.append(precedent)
                self._log(f"✅ Kept [{idx}] {title}")
            else:
                self._log(f"❌ Discarded [{idx}] {title}", "warning")

        self._log(f"📊 Kept {len(relevant)}/{len(precedents)} precedents")
        return relevant

    def _format_context_for_prompt(
        self,
        statutes: list[dict],
        precedents: list[dict],
    ) -> str:
        """Format retrieved context for inclusion in prompts."""
        parts = []

        if statutes:
            parts.append("ARTICOLI DI LEGGE:")
            for s in statutes:
                source = self._source_short_label(s.get("source", ""))
                parts.append(f"- Art. {s.get('articolo')} {source}: {s.get('titolo')}")
                testo = s.get("testo")
                if testo:
                    parts.append(f"  {testo[:settings.truncation_prompt_testo]}...")
            parts.append("")

        if precedents:
            parts.append("PRECEDENTI GIURISPRUDENZIALI:")
            for p in precedents:
                title = p.get("title", "Untitled")
                parts.append(f"- {title}")
                summary = p.get("summary")
                if summary:
                    parts.append(f"  {summary[:settings.truncation_prompt_summary]}...")
            parts.append("")

        return "\n".join(parts) if parts else "No legal context available."

    @staticmethod
    def _source_short_label(source: str) -> str:
        source_map = {
            "codice_civile": "c.c.",
            "codice_penale": "c.p.",
            "codice_amministrativo": "L. 241/1990",
        }
        return source_map.get(source, source or "codice")

    def _norm_to_statute_dict(self, norm: dict) -> dict:
        """
        Convert a taxonomy norm entry into a statute-like dict for prompts.
        Retrieves actual statute text from Neo4j database.
        Supports both old keys (riferimento/nota) and new keys (ref/role).
        """
        from .tools.neo4j_tools import get_statute_by_article_tool

        # Support both old keys (riferimento) and new keys (ref)
        riferimento = norm.get("ref") or norm.get("riferimento", "Art. N/D")
        role = norm.get("role") or norm.get("nota", "")

        articolo_match = None
        try:
            import re

            articolo_match = re.search(r"(\d+)", riferimento)
        except Exception:
            pass
        articolo = articolo_match.group(1) if articolo_match else riferimento

        ref_lower = riferimento.lower()
        if (
            "241/1990" in ref_lower
            or "legge 241" in ref_lower
            or "amministrativ" in ref_lower
        ):
            source = "codice_amministrativo"
        elif "c.c" in ref_lower or "civile" in ref_lower:
            source = "codice_civile"
        else:
            source = "codice_penale"

        # Try to fetch actual statute text from database
        try:
            db_result = get_statute_by_article_tool.invoke(
                {"articolo": articolo, "codice": source}
            )
            if db_result.get("found"):
                return {
                    "statute_id": db_result.get("statute_id", riferimento),
                    "articolo": db_result.get("articolo", articolo),
                    "titolo": db_result.get("titolo", role or riferimento),
                    "testo": db_result.get("testo", ""),
                    "libro": db_result.get("libro", ""),
                    "source": db_result.get("source", source),
                    "role": role,  # Keep the taxonomy role for context
                }
        except Exception as e:
            self._log(f"⚠️ Failed to fetch statute {articolo} from DB: {e}", "warning")

        # Fallback: return dict without actual text
        return {
            "statute_id": riferimento,
            "articolo": articolo,
            "titolo": role or riferimento,
            "testo": f"[Testo art. {articolo} non disponibile nel database]",
            "libro": "",
            "source": source,
            "role": role,
        }
