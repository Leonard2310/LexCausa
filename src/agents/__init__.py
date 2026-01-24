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
    # Polisher-Evaluator
    "PolisherEvaluator",
    "EvaluationResult",
    "ArgumentStatus",
]
