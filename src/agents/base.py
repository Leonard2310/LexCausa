"""
Base Agent class for LexCausa reasoning system.

Provides common functionality for all agents including:
- LLM initialization (Groq)
- Neo4j connection
- Logging and error handling
- Common extraction methods for tool messages
"""

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from neo4j import GraphDatabase

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402


@dataclass
class AgentConfig:
    """Configuration for LexCausa agents."""

    # Groq LLM settings
    groq_api_key: str = field(default_factory=lambda: settings.groq_api_key)
    model_name: str = field(default_factory=lambda: settings.groq_model)
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
        self._neo4j_driver = None

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

    def _rebuild_llm(self, api_key: str, model: str) -> ChatGroq:
        """Rebuild the LLM with a new API key and model (used by resilient wrappers)."""
        return get_chat_groq(
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=api_key,
        )

    def _resilient_llm_invoke(self, messages, **kwargs):
        """Invoke LLM with automatic retry, key rotation, and model fallback."""
        return resilient_chat_call(self.llm, messages, **kwargs)

    @property
    def neo4j_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
        return self._neo4j_driver

    def close(self):
        """Close connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()

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

    # =========================================================================
    # COMMON EXTRACTION METHODS (shared by Reasoner and CounterReasoner)
    # =========================================================================

    def _extract_statutes_from_messages(self, messages: list) -> list[dict]:
        """
        Extract statutes from search_legal_sources or search_statutes tool responses.
        Filters out error/empty messages.
        """
        statutes = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)

                    # Handle search_legal_sources (primary tool - new format)
                    if msg.name == "search_legal_sources":
                        if isinstance(data, dict) and "articles" in data:
                            for item in data["articles"]:
                                if isinstance(item, dict) and "statute_id" in item:
                                    statutes.append(item)

                    # Handle search_statutes (secondary tool - list format)
                    elif msg.name == "search_statutes":
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "statute_id" in item:
                                    statutes.append(item)
                except Exception:
                    pass
        return statutes

    def _extract_precedents_from_messages(self, messages: list) -> list[dict]:
        """
        Extract precedents retrieved by search_precedents tool.
        Filters out error/empty messages.
        """
        precedents = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_precedents":
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "precedent_id" in item:
                                precedents.append(item)
                except Exception:
                    pass
        return precedents

    def _extract_causality_from_messages(self, messages: list) -> dict:
        """
        Extract the result of the classify_causality tool from LangGraph messages.
        """
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "classify_causality":
                try:
                    return json.loads(msg.content)
                except Exception:
                    return {}
        return {}

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

            if in_chain and line:
                # Skip markdown bold section headers (e.g. **Ulteriore Norma**:)
                if line.startswith("**"):
                    continue
                # Numbered items (1., 2., etc.)
                if line[0].isdigit() and len(line) > 2:
                    chain.append(line.lstrip("0123456789.) "))
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
                source = "c.c." if s.get("source") == "codice_civile" else "c.p."
                parts.append(f"- Art. {s.get('articolo')} {source}: {s.get('titolo')}")
                testo = s.get("testo")
                if testo:
                    parts.append(f"  {testo[:500]}...")
            parts.append("")

        if precedents:
            parts.append("PRECEDENTI GIURISPRUDENZIALI:")
            for p in precedents:
                title = p.get("title", "Untitled")
                parts.append(f"- {title}")
                summary = p.get("summary")
                if summary:
                    parts.append(f"  {summary[:300]}...")
            parts.append("")

        return "\n".join(parts) if parts else "No legal context available."

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

        source = "codice_civile" if "c.c" in riferimento.lower() else "codice_penale"

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
