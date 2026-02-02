"""
LangChain Tools for Neo4j Knowledge Base interaction.

Provides tools for agents to search and retrieve:
- Statutes (Codice Civile, Codice Penale)
- Precedents (itacasehold)

Uses LegalSearchPipeline for embedding and vector search - the SAME approach as Tab Ricerca.
"""

import sys
import threading
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.legal_search import LegalSearchPipeline  # noqa: E402

# Singleton driver for tools (only used for direct Neo4j queries like get_statute_by_article)
_driver: Optional[GraphDatabase.driver] = None

# Singleton LegalSearchPipeline (the SAME one that works in Tab Ricerca!)
_legal_search_pipeline: Optional[LegalSearchPipeline] = None
_pipeline_lock = threading.Lock()


def get_driver():
    """Get or create Neo4j driver for direct queries."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def get_legal_search_pipeline() -> LegalSearchPipeline:
    """Get or create LegalSearchPipeline (singleton, thread-safe).

    This is the SAME pipeline that works in Tab Ricerca!
    It handles embedding generation and vector search internally.
    """
    global _legal_search_pipeline

    if _legal_search_pipeline is None:
        with _pipeline_lock:
            if _legal_search_pipeline is None:
                print("🔧 [neo4j_tools] Initializing LegalSearchPipeline...")
                _legal_search_pipeline = LegalSearchPipeline()
                print("✅ [neo4j_tools] LegalSearchPipeline ready!")

    return _legal_search_pipeline


# =============================================================================
# PRIMARY TOOL: search_legal_sources (replicates Tab Ricerca exactly)
# =============================================================================


class SearchLegalSourcesInput(BaseModel):
    """Input schema for the main legal search tool."""

    claim: str = Field(
        description="The COMPLETE original legal claim text. Do NOT rephrase or summarize - use the exact claim!"
    )
    top_k: int = Field(
        default=settings.search_top_k_default,
        description="Maximum number of articles to return (defaults to config SEARCH_TOP_K_DEFAULT)",
    )
    use_top_n_libri: int = Field(
        default=settings.search_use_top_n_libri,
        description="Number of top classified libri (books) to search in (defaults to config SEARCH_USE_TOP_N_LIBRI)",
    )


@tool("search_legal_sources", args_schema=SearchLegalSourcesInput)
def search_legal_sources_tool(
    claim: str,
    top_k: int = settings.search_top_k_default,
    use_top_n_libri: int = settings.search_use_top_n_libri,
) -> dict:
    """
    Search for relevant legal articles using the COMPLETE claim text.

    This is the PRIMARY search tool that replicates Tab Ricerca exactly:
    1. Classifies the claim to identify relevant libri (books)
    2. Generates embedding from the COMPLETE original claim
    3. Performs vector search filtered by the classified libri

    CRITICAL: Pass the EXACT original claim text, do NOT rephrase or summarize it!
    The embedding quality depends on using the full original claim.

    Returns classification info and relevant articles from Civil/Penal Code.
    """
    print(f"🔍 [search_legal_sources] Using COMPLETE claim for search (top_k={top_k})")
    print(f"   Claim preview: '{claim[:100]}...'")

    # Use the SAME pipeline that powers Tab Ricerca
    pipeline = get_legal_search_pipeline()

    # This is EXACTLY what Tab Ricerca does!
    result = pipeline.search(claim, top_k=top_k, use_top_n_libri=use_top_n_libri)

    # Format output for the agent
    articles = []
    for art in result.articles:
        articles.append(
            {
                "statute_id": art.statute_id,
                "articolo": art.articolo,
                "titolo": art.titolo,
                "testo": art.testo[:500] if art.testo else "No text available",
                "libro": art.libro,
                "source": art.source,
                "score": float(art.score),
            }
        )
        source_label = "C.C." if art.source == "codice_civile" else "C.P."
        print(
            f"   📜 Found: Art. {art.articolo} {source_label} - {art.titolo[:40]}... (score: {art.score:.4f})"
        )

    output = {
        "classification": {
            "categories": result.classification.categories,
            "descriptions": result.classification.descriptions,
            "libri": [libro for _, libro in result.classification.libro_mappings],
            "sources": [source for source, _ in result.classification.libro_mappings],
        },
        "articles": articles,
        "total_found": len(articles),
    }

    print(f"   ✅ Total articles found: {len(articles)} [EXACT SAME AS TAB RICERCA]")
    return output


# =============================================================================
# SECONDARY TOOLS: For specific searches when needed
# =============================================================================


class SearchStatutesInput(BaseModel):
    """Input schema for statute search."""

    query: str = Field(description="Search text to find relevant articles")
    codice: str = Field(
        default="both",
        description="Code to search: 'codice_civile', 'codice_penale', or 'both' for all",
    )
    libro: Optional[str] = Field(
        default=None,
        description="Specific book to search (e.g., 'CC Libro V', 'CP Libro II'). If None, searches all.",
    )
    limit: int = Field(
        default=settings.search_top_k_default,
        description="Maximum number of results (defaults to config SEARCH_TOP_K_DEFAULT)",
    )


@tool("search_statutes", args_schema=SearchStatutesInput)
def search_statutes_tool(
    query: str,
    codice: str = "both",
    libro: Optional[str] = None,
    limit: int = settings.search_top_k_default,
) -> list[dict]:
    """
    Search for legal articles in the Italian Civil Code and/or Penal Code using semantic vector search.

    Uses the SAME LegalSearchPipeline that powers the Tab Ricerca.
    Returns relevant articles with title, text, and references.

    IMPORTANT: Always specify the 'libro' parameter (e.g., "CC Libro V") to get relevant results.
    Use classify_claim first to get the correct libro value.
    """
    print(
        f"🔍 [search_statutes] Query: '{query}', codice: {codice}, libro: {libro}, limit: {limit}"
    )

    # Use the SAME pipeline that works in Tab Ricerca!
    pipeline = get_legal_search_pipeline()

    # Generate embedding using the pipeline's method (same as Tab Ricerca)
    print("  🧠 Generating embedding for query...")
    query_embedding = pipeline.embed_text(query)
    print(f"  ✅ Embedding generated (dim: {len(query_embedding)})")

    results = []

    if libro and codice != "both":
        # Use pipeline's vector_search with specific libro filter
        # This is EXACTLY what Tab Ricerca does!
        libri_filters = [(codice, libro)]

        try:
            article_results = pipeline.vector_search(
                query_embedding, libri_filters, top_k=limit
            )

            for article in article_results:
                result_item = {
                    "statute_id": article.statute_id,
                    "articolo": article.articolo,
                    "titolo": article.titolo,
                    "testo": (
                        article.testo[:500] if article.testo else "No text available"
                    ),
                    "libro": article.libro,
                    "source": article.source,
                    "score": float(article.score),
                }
                results.append(result_item)
                source_label = (
                    "C.C." if result_item["source"] == "codice_civile" else "C.P."
                )
                print(
                    f"  📜 Found: Art. {result_item['articolo']} {source_label} - {result_item['titolo'][:40]}... (score: {result_item['score']:.4f})"
                )

        except Exception as e:
            print(f"  ❌ Vector search failed: {str(e)}")
            return _search_statutes_fallback(query, codice, libro, limit)
    else:
        # Fallback: no libro specified, need to do a broader search
        # Use Neo4j driver directly for this case
        driver = get_driver()

        where_clauses = []
        if codice != "both":
            where_clauses.append(f"node.source = '{codice}'")
        if libro:
            where_clauses.append(f"node.libro = '{libro}'")
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query_cypher = f"""
            CALL db.index.vector.queryNodes('statutes_idx', $top_k_expanded, $embedding)
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
        try:
            with driver.session() as session:
                records = session.run(
                    query_cypher,
                    parameters={
                        "embedding": query_embedding,
                        "limit": limit,
                        "top_k_expanded": limit * 20,
                    },
                )
                for record in records:
                    result_item = {
                        "statute_id": record["id"] or "",
                        "articolo": record["articolo"] or "",
                        "titolo": record["titolo"] or "No title",
                        "testo": (
                            record["testo"][:500]
                            if record["testo"]
                            else "No text available"
                        ),
                        "libro": record["libro"] or "",
                        "source": record["source"] or "",
                        "score": (
                            float(record["score"])
                            if record["score"] is not None
                            else 0.0
                        ),
                    }
                    results.append(result_item)
                    source_label = (
                        "C.C." if result_item["source"] == "codice_civile" else "C.P."
                    )
                    print(
                        f"  📜 Found: Art. {result_item['articolo']} {source_label} - {result_item['titolo'][:40]}... (score: {result_item['score']:.4f})"
                    )
        except Exception as e:
            print(f"  ❌ Vector search failed: {str(e)}")
            return _search_statutes_fallback(query, codice, libro, limit)

    if not results:
        print("  ⚠️ No statutes found via vector search, trying fulltext fallback...")
        return _search_statutes_fallback(query, codice, libro, limit)

    print(
        f"  ✅ Total statutes found: {len(results)} [VECTOR SEARCH via LegalSearchPipeline]"
    )
    return results


def _search_statutes_fallback(
    query: str,
    codice: str = "both",
    libro: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Fallback fulltext search for statutes when vector search fails.
    """
    driver = get_driver()

    print(f"  🔄 [fallback] Fulltext search for: '{query}'")

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
            records = session.run(
                query_cypher, parameters={"query": query, "limit": limit}
            )
            for record in records:
                result_item = {
                    "statute_id": record["id"] or "",
                    "articolo": record["articolo"] or "",
                    "titolo": record["titolo"] or "No title",
                    "testo": (
                        record["testo"][:500]
                        if record["testo"]
                        else "No text available"
                    ),
                    "libro": record["libro"] or "",
                    "source": record["source"] or "",
                    "score": (
                        float(record["score"]) if record["score"] is not None else 0.0
                    ),
                }
                results.append(result_item)
                source_label = (
                    "C.C." if result_item["source"] == "codice_civile" else "C.P."
                )
                print(
                    f"  📜 [fallback] Found: Art. {result_item['articolo']} {source_label} (score: {result_item['score']:.2f})"
                )
    except Exception as e:
        print(f"  ❌ Fulltext fallback also failed: {str(e)}")
        return [{"error": f"Search failed: {str(e)}", "query": query}]

    if not results:
        print(f"  ⚠️ No statutes found for query: '{query}'")
        return [{"message": f"No statutes found for query: '{query}'", "query": query}]

    print(f"  ⚠️ Total statutes found: {len(results)} [FALLBACK - FULLTEXT SEARCH]")
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

    # Normalize articolo: database stores as "art1223", input might be "1223" or "art1223"
    articolo_normalized = articolo.lower().replace("art", "").replace(" ", "").strip()
    articolo_with_prefix = f"art{articolo_normalized}"

    query = """
        MATCH (s:Statute)
        WHERE (s.articolo = $articolo OR s.articolo = $articolo_with_prefix) AND s.source = $codice
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
            result = session.run(
                query,
                parameters={
                    "articolo": articolo_normalized,
                    "articolo_with_prefix": articolo_with_prefix,
                    "codice": codice,
                },
            )
            record = result.single()

            if record:
                return {
                    "statute_id": record["id"] or "",
                    "articolo": record["articolo"] or articolo,
                    "titolo": record["titolo"] or "No title",
                    "testo": record["testo"] or "No text available",
                    "libro": record["libro"] or "",
                    "source": record["source"] or codice,
                    "found": True,
                }
            # Return a valid dict even when not found
            return {
                "error": f"Article {articolo} not found in {codice}",
                "articolo": articolo,
                "codice": codice,
                "found": False,
            }
    except Exception as e:
        return {
            "error": f"Query failed: {str(e)}",
            "articolo": articolo,
            "codice": codice,
            "found": False,
        }


class SearchPrecedentsInput(BaseModel):
    """Input schema for precedent search."""

    query: str = Field(description="Search text to find relevant precedents")
    materia: Optional[str] = Field(
        default=None,
        description="Specific subject (e.g., 'civile', 'penale'). If None, searches all.",
    )
    limit: int = Field(
        default=settings.precedents_limit_default,
        description="Maximum number of results (defaults to config PRECEDENTS_LIMIT_DEFAULT)",
    )


@tool("search_precedents", args_schema=SearchPrecedentsInput)
def search_precedents_tool(
    query: str,
    materia: Optional[str] = None,
    limit: int = settings.precedents_limit_default,
) -> list[dict]:
    """
    Search for legal precedents in the Knowledge Base using semantic vector search.

    Returns relevant court decisions and cases with title, summary, and references.
    Use this function when you need to find precedents supporting an argument.
    """
    print(
        f"🔍 [search_precedents] Query: '{query}', materia: {materia}, limit: {limit}"
    )

    # Use the SAME pipeline for embeddings (consistent with Tab Ricerca)
    pipeline = get_legal_search_pipeline()

    print("  🧠 Generating embedding for query...")
    query_embedding = pipeline.embed_text(query)
    print(f"  ✅ Embedding generated (dim: {len(query_embedding)})")

    # Vector similarity search in Neo4j using pipeline's driver
    # Use precedents_idx vector index
    query_cypher = """
        CALL db.index.vector.queryNodes('precedents_idx', $top_k_expanded, $embedding)
        YIELD node, score
        RETURN node.chunk_id AS id,
               node.title AS title,
               node.summary AS summary,
               node.materia AS materia,
               node.url AS url,
               node.chunk_text AS chunk_text,
               score
        ORDER BY score DESC
        LIMIT $limit
    """

    results = []
    try:
        with pipeline.driver.session() as session:
            records = session.run(
                query_cypher,
                parameters={
                    "embedding": query_embedding,
                    "limit": limit,
                    "top_k_expanded": limit * 10,  # Expand for materia filtering
                },
            )
            for record in records:
                # Apply materia filter if specified (post-filter)
                if (
                    materia
                    and record["materia"]
                    and materia.lower() not in record["materia"].lower()
                ):
                    continue

                result_item = {
                    "precedent_id": record["id"] or "",
                    "title": record["title"] or "Untitled precedent",
                    "summary": (
                        record["summary"][:500]
                        if record["summary"]
                        else "No summary available"
                    ),
                    "materia": record["materia"] or "Unknown",
                    "url": record["url"] or "",
                    "excerpt": (
                        record["chunk_text"][:300]
                        if record["chunk_text"]
                        else "No excerpt available"
                    ),
                    "score": (
                        float(record["score"]) if record["score"] is not None else 0.0
                    ),
                }
                results.append(result_item)
                # Debug log each precedent found
                print(
                    f"  📋 Found: {result_item['title'][:60]}... (score: {result_item['score']:.4f})"
                )

    except Exception as e:
        print(f"  ❌ Vector search failed: {str(e)}")
        # Fallback to text search if vector search fails
        print("  🔄 Falling back to text search...")
        return _search_precedents_fallback(query, materia, limit)

    if not results:
        print("  ⚠️ No precedents found via vector search, trying text fallback...")
        return _search_precedents_fallback(query, materia, limit)

    print(f"  ✅ Total precedents found: {len(results)} [VECTOR SEARCH]")
    return results


def _search_precedents_fallback(
    query: str,
    materia: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Fallback text search for precedents when vector search fails.
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
            records = session.run(
                query_cypher, parameters={"query": query, "limit": limit}
            )
            for record in records:
                result_item = {
                    "precedent_id": record["id"] or "",
                    "title": record["title"] or "Untitled precedent",
                    "summary": (
                        record["summary"][:500]
                        if record["summary"]
                        else "No summary available"
                    ),
                    "materia": record["materia"] or "Unknown",
                    "url": record["url"] or "",
                    "excerpt": (
                        record["chunk_text"][:300]
                        if record["chunk_text"]
                        else "No excerpt available"
                    ),
                    "score": record["score"] if record["score"] is not None else 0.0,
                }
                results.append(result_item)
                print(
                    f"  📋 [fallback] Found: {result_item['title'][:60]}... (score: {result_item['score']})"
                )

    except Exception as e:
        print(f"  ❌ Fallback search also failed: {str(e)}")
        return [{"error": f"Search failed: {str(e)}", "query": query}]

    if not results:
        print(f"  ⚠️ No precedents found for query: '{query}'")
        return [
            {"message": f"No precedents found for query: '{query}'", "query": query}
        ]

    print(f"  ⚠️ Total precedents found: {len(results)} [FALLBACK - TEXT SEARCH]")
    return results


def close_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
