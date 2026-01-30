"""
LangChain Tools for LexCausa agents.

Contains tools for:
- Neo4j KB interaction (statutes, precedents)
- Causality taxonomy classification
- Claim classification for libro routing
- Unified legal search (replicates Tab Ricerca)
"""

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
