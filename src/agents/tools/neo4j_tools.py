"""
LangChain Tools for Neo4j Knowledge Base interaction.

Provides tools for agents to search and retrieve:
- Statutes (Codice Civile, Codice Penale, Codice Amministrativo L. 241/1990)
- Precedents (itacasehold)

Uses LegalSearchPipeline for embedding and vector search - the SAME approach as Tab Ricerca.
"""

import sys
import threading
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from neo4j import Driver, GraphDatabase
from pydantic import BaseModel, Field

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.legal_search import LegalSearchPipeline  # noqa: E402

# Singleton driver for tools (only used for direct Neo4j queries like get_statute_by_article)
_driver: Optional[Driver] = None

# Singleton LegalSearchPipeline (the SAME one that works in Tab Ricerca!)
_legal_search_pipeline: Optional[LegalSearchPipeline] = None
_pipeline_lock = threading.Lock()


def _source_short_label(source: object) -> str:
    source_str = str(source or "")
    labels = {
        "codice_civile": "C.C.",
        "codice_penale": "C.P.",
        "codice_amministrativo": "L. 241/1990",
    }
    return labels.get(source_str, source_str or "COD")


def get_driver() -> Driver:
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
                "testo": (
                    art.testo[: settings.truncation_tool_testo]
                    if art.testo
                    else "No text available"
                ),
                "libro": art.libro,
                "source": art.source,
                "score": float(art.score),
            }
        )
        source_label = _source_short_label(art.source)
        print(
            f"   📜 Found: Art. {art.articolo} {source_label} - {art.titolo[:40]}... (score: {art.score:.4f})"
        )

    output = {
        "classification": {
            "categories": result.classification.categories,
            "descriptions": result.classification.descriptions,
            "libri": [
                libro or "(intero codice, senza libri)"
                for _, libro in result.classification.libro_mappings
            ],
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
        description="Code to search: 'codice_civile', 'codice_penale', 'codice_amministrativo', or 'both' for all",
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
    Search for legal articles in Italian Civil/Penal/Administrative codes using semantic vector search.

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
                        article.testo[: settings.truncation_tool_testo]
                        if article.testo
                        else "No text available"
                    ),
                    "libro": article.libro,
                    "source": article.source,
                    "score": float(article.score),
                }
                results.append(result_item)
                source_label = _source_short_label(result_item["source"])
                print(
                    f"  📜 Found: Art. {result_item['articolo']} {source_label} - {result_item['titolo'][:40]}... (score: {result_item['score']:.4f})"
                )

        except Exception as e:
            print(f"  ❌ Vector search failed: {str(e)}")
            print("  ⚠️ Switching to fulltext fallback...")
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
                            record["testo"][: settings.truncation_tool_testo]
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
                    source_label = _source_short_label(result_item["source"])
                    print(
                        f"  📜 Found: Art. {result_item['articolo']} {source_label} - {result_item['titolo'][:40]}... (score: {result_item['score']:.4f})"
                    )
        except Exception as e:
            print(f"  ❌ Vector search failed: {str(e)}")
            print("  ⚠️ Switching to fulltext fallback...")
            return _search_statutes_fallback(query, codice, libro, limit)

    if not results:
        print(
            "  ⚠️ No statutes found via vector search, switching to fulltext fallback..."
        )
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
    Fulltext search for statutes.

    Used both as fallback when vector search fails and as primary method
    for cross-reference reverse lookups (where fulltext is the correct strategy).
    """
    driver = get_driver()

    print(f"  🔄 Fulltext search for: '{query}'")

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
                        record["testo"][: settings.truncation_tool_testo]
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
                source_label = _source_short_label(result_item["source"])
                print(
                    f"  📜 Found: Art. {result_item['articolo']} {source_label} (score: {result_item['score']:.2f})"
                )
    except Exception as e:
        print(f"  ❌ Fulltext search failed: {str(e)}")
        return [{"error": f"Search failed: {str(e)}", "query": query}]

    if not results:
        print(f"  ⚠️ No statutes found for query: '{query}'")
        return [{"message": f"No statutes found for query: '{query}'", "query": query}]

    print(f"  ✅ Total statutes found: {len(results)} [FULLTEXT SEARCH]")
    return results


class GetStatuteByArticleInput(BaseModel):
    """Input schema for getting statute by article number."""

    articolo: str = Field(description="Article number (e.g., '2043', '40')")
    codice: str = Field(
        description="Reference code: 'codice_civile', 'codice_penale', or 'codice_amministrativo'"
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

    query: str = Field(description="The legal claim to extract keywords from")
    limit: int = Field(
        default=settings.precedents_limit_default,
        description="Maximum number of results (defaults to config PRECEDENTS_LIMIT_DEFAULT)",
    )


def _extract_keywords_from_claim(claim: str) -> list[str]:
    """Use LLM to extract legal keywords from a claim for precedent search.

    Returns a list of keyword strings suitable for Neo4j fulltext queries.
    """
    from langchain_core.messages import HumanMessage

    from services.groq_client import get_chat_groq, resilient_chat_call

    prompt = f"""Extract the most important legal keywords from this claim.
Focus on: legal concepts, legal domains, types of offenses/violations,
key factual elements, and relevant legal categories.

CLAIM:
"{claim}"

RULES:
- Extract 5 to 10 keywords or short phrases (max 3 words each).
- Use Italian legal terminology.
- One keyword per line.
- Do NOT add numbering, bullets, or explanations.
- Do NOT repeat the claim.
- Output ONLY the keywords, nothing else.
"""

    try:
        llm = get_chat_groq(
            model=settings.groq_model,
            temperature=0.0,
            max_tokens=200,
            api_key=settings.groq_api_key or None,
        )
        response = resilient_chat_call(llm, [HumanMessage(content=prompt)])
        raw = response.content.strip()
        keywords = [
            kw.strip().strip("-•*").strip()
            for kw in raw.split("\n")
            if kw.strip() and len(kw.strip()) > 1
        ]
        # Deduplicate preserving order
        seen = set()
        unique = []
        for kw in keywords:
            low = kw.lower()
            if low not in seen:
                seen.add(low)
                unique.append(kw)
        print(f"  🔑 Extracted keywords: {unique}")
        return unique[:10]
    except Exception as e:
        print(f"  ⚠️ Keyword extraction failed: {e}")
        # Fallback: split claim into meaningful chunks
        import re

        words = re.findall(r"\b[a-zA-ZàèéìòùÀÈÉÌÒÙ]{4,}\b", claim)
        return list(dict.fromkeys(words))[:8]


def _search_precedents_by_keywords(
    keywords: list[str],
    limit: int = 5,
) -> list[dict]:
    """Search precedents via Neo4j fulltext index using extracted keywords.

    Builds a Lucene OR query from the keywords and queries the
    ``precedents_fulltext_idx`` index (on ``title`` and ``summary``).
    """
    driver = get_driver()

    # Build Lucene query: each keyword OR'd together
    lucene_query = " OR ".join(keywords)
    print(f"  🔄 Fulltext search with query: '{lucene_query}'")

    query_cypher = """
        CALL db.index.fulltext.queryNodes('precedents_fulltext_idx', $query)
        YIELD node, score
        RETURN node.precedent_id AS id,
               node.title AS title,
               node.summary AS summary,
               node.url AS url,
               score
        ORDER BY score DESC
        LIMIT $limit
    """

    results: list[dict[str, str | float]] = []
    try:
        with driver.session() as session:
            records = session.run(
                query_cypher,
                parameters={"query": lucene_query, "limit": limit},
            )
            for record in records:
                title = str(record["title"] or "Untitled precedent")
                summary_val = record["summary"]
                if isinstance(summary_val, str):
                    summary = (
                        summary_val[: settings.truncation_tool_summary]
                        if summary_val
                        else "No summary available"
                    )
                else:
                    summary = (
                        str(summary_val)[: settings.truncation_tool_summary]
                        if summary_val is not None
                        else "No summary available"
                    )
                result_item = {
                    "precedent_id": record["id"] or "",
                    "title": title,
                    "summary": summary,
                    "url": record["url"] or "",
                    "score": (
                        float(record["score"]) if record["score"] is not None else 0.0
                    ),
                }
                results.append(result_item)
                print(
                    f"  📋 Found: {title[:60]}... "
                    f"(score: {result_item['score']:.4f})"
                )
    except Exception as e:
        print(f"  ❌ Fulltext search failed: {str(e)}")
        return []

    return results


@tool("search_precedents", args_schema=SearchPrecedentsInput)
def search_precedents_tool(
    query: str,
    limit: int = settings.precedents_limit_default,
) -> list[dict]:
    """
    Search for legal precedents using keyword extraction + fulltext search.

    1. Extracts legal keywords from the claim via LLM.
    2. Queries Neo4j fulltext index (precedents_fulltext_idx) with those keywords.
    3. Returns matching precedents sorted by relevance score.
    """
    print(f"🔍 [search_precedents] Claim: '{query[:80]}...', limit: {limit}")

    # Step 1: Extract keywords from the claim
    keywords = _extract_keywords_from_claim(query)
    if not keywords:
        print("  ⚠️ No keywords extracted, returning empty results")
        return []

    # Step 2: Fulltext search with keywords
    results = _search_precedents_by_keywords(keywords, limit=limit)

    if not results:
        print("  ⚠️ No precedents found via keyword search")
        return []

    print(f"  ✅ Total precedents found: {len(results)} [KEYWORD + FULLTEXT]")
    return results
