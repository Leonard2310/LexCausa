"""
LangChain Tools for Neo4j Knowledge Base interaction.

Provides tools for agents to search and retrieve:
- Statutes (Codice Civile, Codice Penale)
- Precedents (itacasehold)
"""

import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402

# Singleton driver for tools
_driver: Optional[GraphDatabase.driver] = None


def get_driver():
    """Get or create Neo4j driver."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


class SearchStatutesInput(BaseModel):
    """Input schema for statute search."""

    query: str = Field(description="Search text to find relevant articles")
    codice: str = Field(
        default="both",
        description="Code to search: 'codice_civile', 'codice_penale', or 'both' for all",
    )
    libro: Optional[str] = Field(
        default=None,
        description="Specific book to search (e.g., 'CC Libro IV', 'CP Libro II'). If None, searches all.",
    )
    limit: int = Field(default=5, description="Maximum number of results")


@tool("search_statutes", args_schema=SearchStatutesInput)
def search_statutes_tool(
    query: str,
    codice: str = "both",
    libro: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Search for legal articles in the Italian Civil Code and/or Penal Code using fulltext search.

    Returns relevant articles with title, text, and references.
    Use this function when you need to find norms relevant to a legal issue.
    """
    driver = get_driver()

    # Build WHERE clause
    where_clauses = []
    if codice != "both":
        where_clauses.append(f"node.source = '{codice}'")
    if libro:
        where_clauses.append(f"node.libro = '{libro}'")

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query_cypher = f"""
        CALL db.index.fulltext.queryNodes('statutes_fulltext_idx', $query)
        YIELD node, score
        {where_clause}
        RETURN node.statute_id AS id,
               node.articolo AS articolo,
               node.titolo AS titolo,
               node.testo AS testo,
               node.libro AS libro,
               node.source AS source,
               score
        ORDER BY score DESC
        LIMIT $limit
    """

    results = []
    try:
        with driver.session() as session:
            records = session.run(query_cypher, parameters={"query": query, "limit": limit})
            for record in records:
                results.append(
                    {
                        "statute_id": record["id"] or "",
                        "articolo": record["articolo"] or "",
                        "titolo": record["titolo"] or "No title",
                        "testo": record["testo"][:500] if record["testo"] else "No text available",
                        "libro": record["libro"] or "",
                        "source": record["source"] or "",
                        "score": record["score"] if record["score"] is not None else 0.0,
                    }
                )
    except Exception as e:
        # Return error as a valid result
        return [{"error": f"Search failed: {str(e)}", "query": query}]

    # Always return at least one item
    if not results:
        return [{"message": f"No statutes found for query: '{query}'", "query": query}]
    
    return results


class GetStatuteByArticleInput(BaseModel):
    """Input schema for getting statute by article number."""

    articolo: str = Field(description="Article number (e.g., '2043', '40')")
    codice: str = Field(
        description="Reference code: 'codice_civile' or 'codice_penale'"
    )


@tool("get_statute_by_article", args_schema=GetStatuteByArticleInput)
def get_statute_by_article_tool(articolo: str, codice: str) -> dict:
    """
    Retrieve a specific article given its number and code.

    Use this function when you know the exact legal reference (e.g., Art. 2043 c.c.).
    """
    driver = get_driver()

    query = """
        MATCH (s:Statute)
        WHERE s.articolo = $articolo AND s.source = $codice
        RETURN s.statute_id AS id,
               s.articolo AS articolo,
               s.titolo AS titolo,
               s.testo AS testo,
               s.libro AS libro,
               s.source AS source
        LIMIT 1
    """

    try:
        with driver.session() as session:
            result = session.run(query, parameters={"articolo": articolo, "codice": codice})
            record = result.single()

            if record:
                return {
                    "statute_id": record["id"] or "",
                    "articolo": record["articolo"] or articolo,
                    "titolo": record["titolo"] or "No title",
                    "testo": record["testo"] or "No text available",
                    "libro": record["libro"] or "",
                    "source": record["source"] or codice,
                    "found": True
                }
            # Return a valid dict even when not found
            return {
                "error": f"Article {articolo} not found in {codice}",
                "articolo": articolo,
                "codice": codice,
                "found": False
            }
    except Exception as e:
        return {
            "error": f"Query failed: {str(e)}",
            "articolo": articolo,
            "codice": codice,
            "found": False
        }


class SearchPrecedentsInput(BaseModel):
    """Input schema for precedent search."""

    query: str = Field(description="Search text to find relevant precedents")
    materia: Optional[str] = Field(
        default=None,
        description="Specific subject (e.g., 'civile', 'penale'). If None, searches all.",
    )
    limit: int = Field(default=5, description="Maximum number of results")


@tool("search_precedents", args_schema=SearchPrecedentsInput)
def search_precedents_tool(
    query: str,
    materia: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Search for legal precedents in the Knowledge Base.

    Returns relevant court decisions and cases with title, summary, and references.
    Use this function when you need to find precedents supporting an argument.
    """
    driver = get_driver()

    where_clause = ""
    if materia:
        where_clause = f"WHERE p.materia = '{materia}'"

    query_cypher = f"""
        MATCH (p:Precedent)
        {where_clause}
        WITH p,
             CASE
                WHEN toLower(p.title) CONTAINS toLower($query) THEN 2.0
                WHEN toLower(p.summary) CONTAINS toLower($query) THEN 1.5
                WHEN toLower(p.chunk_text) CONTAINS toLower($query) THEN 1.0
                ELSE 0.0
             END AS score
        WHERE score > 0
        RETURN p.chunk_id AS id,
               p.title AS title,
               p.summary AS summary,
               p.materia AS materia,
               p.url AS url,
               p.chunk_text AS chunk_text,
               score
        ORDER BY score DESC
        LIMIT $limit
    """

    results = []
    try:
        with driver.session() as session:
            records = session.run(query_cypher, parameters={"query": query, "limit": limit})
            for record in records:
                results.append(
                    {
                        "precedent_id": record["id"] or "",
                        "title": record["title"] or "Untitled precedent",
                        "summary": record["summary"][:300] if record["summary"] else "No summary available",
                        "materia": record["materia"] or "Unknown",
                        "url": record["url"] or "",
                        "excerpt": (
                            record["chunk_text"][:200] if record["chunk_text"] else "No excerpt available"
                        ),
                        "score": record["score"] if record["score"] is not None else 0.0,
                    }
                )
    except Exception as e:
        # Return error as a valid result
        return [{"error": f"Search failed: {str(e)}", "query": query}]

    # Always return at least one item
    if not results:
        return [{"message": f"No precedents found for query: '{query}'", "query": query}]
    
    return results


def close_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None