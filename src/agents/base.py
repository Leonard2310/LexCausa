"""
Base Agent class for LexCausa reasoning system.

Provides common functionality for all agents including:
- LLM initialization (Groq)
- Neo4j connection
- Logging and error handling
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_groq import ChatGroq
from neo4j import GraphDatabase

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402


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
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = ChatGroq(
                api_key=self.config.groq_api_key,
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        return self._llm

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
