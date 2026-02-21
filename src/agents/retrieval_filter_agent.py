"""
Retrieval Filter Agent for LexCausa.

Agent tecnico usato esclusivamente in fase di retrieval per:
- estrazione contesto legale del claim
- filtro rilevanza norme/precedenti
- filtro applicabilità norme

Non esegue ragionamento principale/counter.
"""

from .base import BaseAgent


class RetrievalFilterAgent(BaseAgent):
    """Agent leggero per filtri retrieval con prefisso log dedicato."""

    def run(self, claim: str, *args, **kwargs):
        raise NotImplementedError(
            "RetrievalFilterAgent does not execute reasoning runs"
        )

    def _log(self, message: str, level: str = "info"):
        emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(
            level, "•"
        )
        print(f"{emoji} [Retrieval] {message}")
