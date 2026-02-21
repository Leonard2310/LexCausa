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

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

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
        max_item_logs = max(0, settings.search_filter_log_top_n)

        for idx, statute in enumerate(statutes, start=1):
            article_number = statute.get("articolo", "N/A")
            article_title = statute.get("titolo", "Untitled")
            article_desc = statute.get("testo", "Untitled")

            prompt = f"""Legal Claim:
"{claim}"

Article:
"{article_number} - {article_title} - {article_desc}"

Instruction:
Determine whether the main topic of the article is directly mentioned or implied in the claim.

Rules:
- Do NOT evaluate whether the article fully resolves the issue.
- Do NOT suggest any additional articles.
- Do NOT use external knowledge; only consider the claim and this article.
- Do NOT add explanations or comments.
- Answer YES in all cases with even indirect connection.
- Use NO only when the article is clearly about a different domain.
- If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.
No punctuation. No new lines. No extra spaces.
"""

            try:
                response = self._resilient_llm_invoke([HumanMessage(content=prompt)])
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

            should_log_item = idx <= max_item_logs
            if keep:
                relevant_statutes.append(statute)
                if should_log_item:
                    self._log(
                        f"✅ Keeping article [{idx}] {article_number} - {article_title}"
                    )
            else:
                if should_log_item:
                    self._log(
                        f"❌ Discarding article [{idx}] {article_number} - {article_title}",
                        "warning",
                    )

        if len(statutes) > max_item_logs:
            self._log(
                f"… per-item filter logs truncated: {len(statutes) - max_item_logs} articoli omessi "
                f"(config SEARCH_FILTER_LOG_TOP_N={max_item_logs})",
                "info",
            )

        self._log(f"📊 Result: {len(relevant_statutes)}/{len(statutes)} statutes kept")
        return relevant_statutes

    def _extract_legal_context(self, claim: str) -> str:
        """Extract a short legal context label used by applicability filtering.

        Returns a compact line containing domain/party relationship/procedural
        posture so statute applicability can be evaluated more strictly.
        """
        prompt = f"""You are a legal triage assistant.

Given this claim:
"{claim}"

Extract a compact legal context string (max 20 words) including:
- legal domain (criminal/civil/administrative/labour/commercial/etc.)
- party relationship (private-private, citizen-state, company-shareholders, etc.)
- procedural posture (investigation/trial/enforcement/contract dispute/etc.)

If uncertain, provide the most plausible generic context.

Respond with EXACTLY one short line and no extra text."""

        try:
            response = self._resilient_llm_invoke([HumanMessage(content=prompt)])
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
        max_item_logs = max(0, settings.search_filter_log_top_n)

        for idx, statute in enumerate(statutes, start=1):
            article_number = statute.get("articolo", "N/A")
            article_title = statute.get("titolo", "Untitled")
            article_text = statute.get("testo", "") or ""

            prompt = f"""Legal Situation:
"{claim}"

Legal Context: {legal_context}

Statute:
"{article_number} - {article_title}"
"{article_text[:500]}"

Question:
Does this statute APPLY to the legal situation described?

Evaluation Criteria:
1. Subject Scope: Does the statute apply to the TYPE of parties involved?
2. Substantive Scope: Does the statute regulate the LEGAL ISSUE at stake?
3. Temporal Scope: Is the statute relevant to the PROCEDURAL PHASE?

Rules:
- Answer YES only if the statute directly regulates THIS situation.
- Answer NO if it applies to a different:
  * relationship type
  * legal domain
  * offense class
  * procedural phase
- If uncertain but potentially on-point, answer YES.

Respond with EXACTLY one token: YES or NO."""

            try:
                response = self._resilient_llm_invoke([HumanMessage(content=prompt)])
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

            should_log_item = idx <= max_item_logs
            if keep:
                applicable_statutes.append(statute)
                if should_log_item:
                    self._log(
                        f"✅ APPLICABLE [{idx}] {article_number} - {article_title}"
                    )
            else:
                if should_log_item:
                    self._log(
                        f"❌ NOT APPLICABLE [{idx}] {article_number} - {article_title}",
                        "warning",
                    )

        if len(statutes) > max_item_logs:
            self._log(
                f"… per-item applicability logs truncated: {len(statutes) - max_item_logs} articoli omessi "
                f"(config SEARCH_FILTER_LOG_TOP_N={max_item_logs})",
                "info",
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
        max_item_logs = max(0, settings.search_filter_log_top_n)

        for idx, precedent in enumerate(precedents, start=1):
            title = precedent.get("title", "Untitled")
            summary = precedent.get("summary", "")
            materia = precedent.get("materia", "")

            materia_line = f'\nDomain: "{materia}"' if materia else ""

            prompt = f"""You are a senior Italian legal expert.

CLAIM (the legal case under evaluation):
"{claim}"

PRECEDENT:
Title: "{title}"{materia_line}
Summary: "{summary[:600]}"

TASK — Decide whether a competent lawyer would cite this precedent
when arguing the above claim (either to support or to counter it).

Answer YES when ANY of the following is true:
1. The precedent addresses the SAME or a closely analogous legal
   question (e.g. same offence, same cause of action, same defence).
2. The precedent establishes a legal PRINCIPLE (causation test,
   evidentiary standard, constitutional interpretation, procedural
   rule) that directly applies to the claim.
3. The factual scenario of the precedent is substantially similar to
   the claim, making the ruling transferable.

Answer NO when:
- The precedent concerns a completely unrelated area of law with no
  transferable principle (e.g. tax evasion vs. divorce).
- The connection is merely superficial (shared keywords but different
  legal substance).

If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO."""

            try:
                response = self._resilient_llm_invoke([HumanMessage(content=prompt)])
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

            should_log_item = idx <= max_item_logs
            if keep:
                relevant.append(precedent)
                if should_log_item:
                    self._log(f"✅ Kept [{idx}] {title}")
            else:
                if should_log_item:
                    self._log(f"❌ Discarded [{idx}] {title}", "warning")

        if len(precedents) > max_item_logs:
            self._log(
                f"… per-item precedent logs truncated: {len(precedents) - max_item_logs} elementi omessi "
                f"(config SEARCH_FILTER_LOG_TOP_N={max_item_logs})",
                "info",
            )

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
