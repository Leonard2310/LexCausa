"""LangChain tools for LexCausa agents (lazy exports to avoid import cycles)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .neo4j_tools import (
        get_statute_by_article_tool,
        search_legal_sources_tool,
        search_precedents_tool,
        search_statutes_tool,
    )
    from .taxonomy_tools import classify_causality_tool, get_causality_theory_tool

__all__ = [
    # Primary search tool (replicates Tab Ricerca exactly)
    "search_legal_sources_tool",
    # Secondary tools for specific queries
    "search_statutes_tool",
    "search_precedents_tool",
    "get_statute_by_article_tool",
    # Classification tools
    "classify_causality_tool",
    "get_causality_theory_tool",
]


def __getattr__(name: str) -> Any:
    if name in {
        "get_statute_by_article_tool",
        "search_legal_sources_tool",
        "search_precedents_tool",
        "search_statutes_tool",
    }:
        from . import neo4j_tools

        return getattr(neo4j_tools, name)
    if name in {"classify_causality_tool", "get_causality_theory_tool"}:
        from . import taxonomy_tools

        return getattr(taxonomy_tools, name)
    raise AttributeError(name)
