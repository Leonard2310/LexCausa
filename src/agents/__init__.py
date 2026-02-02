"""
LexCausa Agents Module.

Questo modulo contiene gli agenti LangChain per il ragionamento legale:
- Reasoner: Genera argomenti a supporto del claim
- CounterReasoner: Genera contro-argomenti basati sul tipo di causalità
- PolisherEvaluator: Valuta e raffina gli argomenti generati
"""

from .base import AgentConfig, BaseAgent, ReasoningResult
from .counter_reasoner import CounterArgument, CounterReasoner, CounterReasonerOutput
from .polisher_evaluator import ArgumentStatus, EvaluationResult, PolisherEvaluator
from .reasoner import Reasoner, ReasonerOutput
from .router import Router, RoutingDecision

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "ReasoningResult",
    # Reasoner
    "Reasoner",
    "ReasonerOutput",
    # Counter-Reasoner
    "CounterReasoner",
    "CounterReasonerOutput",
    "CounterArgument",
    # Router
    "Router",
    "RoutingDecision",
    # Polisher-Evaluator
    "PolisherEvaluator",
    "EvaluationResult",
    "ArgumentStatus",
]
