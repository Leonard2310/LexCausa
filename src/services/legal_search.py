"""
Legal Search Pipeline for LexCausa.

Complete pipeline that:
1. Classifies a legal claim using Groq Cloud LLM
2. Generates embedding for the claim using Legal-BERT
3. Performs vector search in Neo4j filtered by relevant libri/source
4. Returns the most relevant articles from Italian law codes
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from neo4j import GraphDatabase
from transformers import AutoModel, AutoTokenizer

from .claim_classifier import ClaimClassifier, ClassificationResult

# Cross-platform path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402


def source_human_label(source: str) -> str:
    source_map = {
        "codice_civile": "Codice Civile",
        "codice_penale": "Codice Penale",
        "codice_amministrativo": "Codice Amministrativo (L. 241/1990)",
    }
    return source_map.get(source, source or "Codice")


@dataclass
class ArticleResult:
    """A single article result from vector search."""

    statute_id: str
    articolo: str
    titolo: str
    testo: str
    libro: str
    source: str
    score: float
    score_debug: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        source_label = source_human_label(self.source)
        lines = [
            f"[{source_label}] Art. {self.articolo} - {self.titolo}",
            f"  Score: {self.score:.4f}",
        ]
        if self.libro:
            lines.insert(1, f"  Libro: {self.libro}")
        lines.append(f"  Testo: {self.testo[:200]}...")
        return "\n".join(lines)


@dataclass
class SearchResult:
    """Complete search result with classification and articles."""

    claim: str
    classification: ClassificationResult
    articles: list[ArticleResult] = field(default_factory=list)
    embedding_model: str = field(default_factory=lambda: settings.embedding_model)

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            "LEXCAUSA - Legal Search Result",
            "=" * 70,
            "",
            f"📝 Claim: {self.claim}",
            "",
            "📊 Classification (Top 3 libri):",
        ]

        for i, (cat, desc) in enumerate(
            zip(
                self.classification.categories,
                self.classification.descriptions,
            ),
            1,
        ):
            lines.append(f"   {i}. {cat} -> {desc}")

        lines.extend(["", f"📚 Found {len(self.articles)} relevant articles:", ""])

        for i, article in enumerate(self.articles, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(str(article))
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


class LegalSearchPipeline:
    """
    Complete pipeline for legal claim search.

    Combines LLM classification with vector similarity search
    to find relevant articles in Italian law codes.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the search pipeline.

        Args:
            groq_api_key: API key for Groq Cloud. If None, reads from settings.
            device: Device for embeddings ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Initialize classifier
        self.classifier = ClaimClassifier(api_key=groq_api_key)

        # Initialize embedding model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading embedding model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.embedding_model, local_files_only=True
        )
        # Use local cached model (pre-downloaded) with weights_only=False
        # to avoid network calls and torch.load security check
        import os

        os.environ["TRUST_REMOTE_CODE"] = "1"
        self.model = AutoModel.from_pretrained(
            settings.embedding_model, local_files_only=True, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"✅ Embedding model loaded: {settings.embedding_model}")

        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        print("✅ Neo4j connection established")

    def close(self):
        """Close connections."""
        self.driver.close()

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embedding."""
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a text using Legal-BERT.

        Args:
            text: The text to embed.

        Returns:
            768-dimensional embedding vector.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=settings.embedding_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        return embedding.cpu().numpy()[0].tolist()

    def vector_search(
        self,
        embedding: list[float],
        libri_filters: list[tuple[str, str]],
        top_k: int = settings.search_top_k_default,
        query_text: Optional[str] = None,
    ) -> list[ArticleResult]:
        """
        Perform hybrid retrieval (vector + fulltext) filtered by source/libro.

        Args:
            embedding: Query embedding vector.
            libri_filters: List of (source, libro) tuples to filter by.
            top_k: Number of results per libro.
            query_text: Optional query text for fulltext branch.

        Returns:
            List of ArticleResult sorted by score.
        """
        all_results: list[ArticleResult] = []
        query_text = (query_text or "").strip()

        with self.driver.session() as session:
            for filter_rank, (source, libro) in enumerate(libri_filters, start=1):
                if not source or source == "unknown":
                    continue

                # For codici without libri (e.g. amministrativo), force source-only.
                libro_filter = (libro or "").strip()
                if source == "codice_amministrativo":
                    libro_filter = ""

                candidate_limit = max(
                    top_k * settings.search_hybrid_candidate_multiplier,
                    settings.search_hybrid_candidate_min,
                )
                vector_results = self._vector_exact_search(
                    session=session,
                    embedding=embedding,
                    source=source,
                    libro=libro_filter,
                    limit=candidate_limit,
                )
                fulltext_results = self._fulltext_search(
                    session=session,
                    query_text=query_text,
                    source=source,
                    libro=libro_filter,
                    limit=candidate_limit,
                )
                fused_results = self._fuse_ranked_results(
                    source=source,
                    vector_results=vector_results,
                    fulltext_results=fulltext_results,
                    query_text=query_text,
                    limit=max(
                        top_k * settings.search_hybrid_fused_pool_multiplier,
                        top_k,
                    ),
                )
                # Preserve classifier ordering signal: top mapped filter has higher
                # priority than secondary/tertiary mappings.
                filter_priority = max(
                    settings.search_hybrid_filter_priority_floor,
                    1.0
                    - (
                        (filter_rank - 1) * settings.search_hybrid_filter_priority_decay
                    ),
                )
                for item in fused_results:
                    item.score *= filter_priority
                    item.score_debug["priority_multiplier"] = float(filter_priority)
                    item.score_debug["final_score"] = float(item.score)
                all_results.extend(fused_results)

        # Global dedupe across filters: keep the best fused score per statute.
        best_by_id: dict[str, ArticleResult] = {}
        for result in all_results:
            existing = best_by_id.get(result.statute_id)
            if existing is None or result.score > existing.score:
                best_by_id[result.statute_id] = result

        return sorted(best_by_id.values(), key=lambda x: x.score, reverse=True)[:top_k]

    def _vector_exact_search(
        self,
        session,
        embedding: list[float],
        source: str,
        libro: str,
        limit: int,
    ) -> list[ArticleResult]:
        """Exact cosine search in Neo4j constrained by source/libro."""
        if libro:
            query = """
            MATCH (node:Statute)
            WHERE node.source = $source
              AND node.libro = $libro
              AND node.embedding IS NOT NULL
            WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
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
            records = session.run(
                query,
                source=source,
                libro=libro,
                embedding=embedding,
                limit=limit,
            )
        else:
            query = """
            MATCH (node:Statute)
            WHERE node.source = $source
              AND node.embedding IS NOT NULL
            WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
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
            records = session.run(
                query,
                source=source,
                embedding=embedding,
                limit=limit,
            )

        return [
            ArticleResult(
                statute_id=record["id"],
                articolo=record["articolo"],
                titolo=record["titolo"],
                testo=record["testo"] or "",
                libro=record["libro"] or "",
                source=record["source"],
                score=float(record["score"] or 0.0),
            )
            for record in records
        ]

    def _fulltext_search(
        self,
        session,
        query_text: str,
        source: str,
        libro: str,
        limit: int,
    ) -> list[ArticleResult]:
        """Fulltext search constrained by source/libro."""
        if not query_text:
            return []

        if libro:
            query = """
            CALL db.index.fulltext.queryNodes('statutes_fulltext_idx', $query_text)
            YIELD node, score
            WHERE node.source = $source AND node.libro = $libro
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
            records = session.run(
                query,
                query_text=query_text,
                source=source,
                libro=libro,
                limit=limit,
            )
        else:
            query = """
            CALL db.index.fulltext.queryNodes('statutes_fulltext_idx', $query_text)
            YIELD node, score
            WHERE node.source = $source
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
            records = session.run(
                query,
                query_text=query_text,
                source=source,
                limit=limit,
            )

        return [
            ArticleResult(
                statute_id=record["id"],
                articolo=record["articolo"],
                titolo=record["titolo"],
                testo=record["testo"] or "",
                libro=record["libro"] or "",
                source=record["source"],
                score=float(record["score"] or 0.0),
            )
            for record in records
        ]

    @staticmethod
    def _rank_score(rank: int, total: int) -> float:
        """Convert rank position to normalized score in [0, 1]."""
        if total <= 1:
            return 1.0
        return 1.0 - ((rank - 1) / (total - 1))

    def _fuse_ranked_results(
        self,
        source: str,
        vector_results: list[ArticleResult],
        fulltext_results: list[ArticleResult],
        query_text: str,
        limit: int,
    ) -> list[ArticleResult]:
        """Fuse vector and fulltext ranked lists with weighted rank scores."""
        if source == "codice_amministrativo":
            vector_weight = settings.search_hybrid_admin_vector_weight
            fulltext_weight = settings.search_hybrid_admin_fulltext_weight
        else:
            vector_weight = settings.search_hybrid_vector_weight
            fulltext_weight = settings.search_hybrid_fulltext_weight

        if not vector_results and not fulltext_results:
            return []
        query_terms = self._extract_query_terms(query_text)
        if not vector_results:
            total = len(fulltext_results)
            return [
                ArticleResult(
                    statute_id=item.statute_id,
                    articolo=item.articolo,
                    titolo=item.titolo,
                    testo=item.testo,
                    libro=item.libro,
                    source=item.source,
                    score=rank_score + keyword_bonus,
                    score_debug={
                        "vector_rank_score": 0.0,
                        "fulltext_rank_score": rank_score,
                        "fusion_score": rank_score,
                        "keyword_bonus": keyword_bonus,
                        "priority_multiplier": 1.0,
                        "final_score": rank_score + keyword_bonus,
                    },
                )
                for idx, item in enumerate(fulltext_results[:limit], start=1)
                for rank_score in [self._rank_score(idx, total)]
                for keyword_bonus in [self._keyword_overlap_bonus(item, query_terms)]
            ]
        if not fulltext_results:
            total = len(vector_results)
            return [
                ArticleResult(
                    statute_id=item.statute_id,
                    articolo=item.articolo,
                    titolo=item.titolo,
                    testo=item.testo,
                    libro=item.libro,
                    source=item.source,
                    score=rank_score + keyword_bonus,
                    score_debug={
                        "vector_rank_score": rank_score,
                        "fulltext_rank_score": 0.0,
                        "fusion_score": rank_score,
                        "keyword_bonus": keyword_bonus,
                        "priority_multiplier": 1.0,
                        "final_score": rank_score + keyword_bonus,
                    },
                )
                for idx, item in enumerate(vector_results[:limit], start=1)
                for rank_score in [self._rank_score(idx, total)]
                for keyword_bonus in [self._keyword_overlap_bonus(item, query_terms)]
            ]

        by_id: dict[str, ArticleResult] = {}
        vector_rank: dict[str, float] = {}
        fulltext_rank: dict[str, float] = {}

        for idx, item in enumerate(vector_results, start=1):
            by_id.setdefault(item.statute_id, item)
            vector_rank[item.statute_id] = self._rank_score(idx, len(vector_results))
        for idx, item in enumerate(fulltext_results, start=1):
            by_id.setdefault(item.statute_id, item)
            fulltext_rank[item.statute_id] = self._rank_score(
                idx, len(fulltext_results)
            )

        fused: list[ArticleResult] = []
        for statute_id, base_item in by_id.items():
            v_score = vector_rank.get(statute_id, 0.0)
            f_score = fulltext_rank.get(statute_id, 0.0)
            fusion_score = (vector_weight * v_score) + (fulltext_weight * f_score)
            keyword_bonus = self._keyword_overlap_bonus(base_item, query_terms)
            score = fusion_score + keyword_bonus
            fused.append(
                ArticleResult(
                    statute_id=base_item.statute_id,
                    articolo=base_item.articolo,
                    titolo=base_item.titolo,
                    testo=base_item.testo,
                    libro=base_item.libro,
                    source=base_item.source,
                    score=score,
                    score_debug={
                        "vector_rank_score": v_score,
                        "fulltext_rank_score": f_score,
                        "fusion_score": fusion_score,
                        "keyword_bonus": keyword_bonus,
                        "priority_multiplier": 1.0,
                        "final_score": score,
                    },
                )
            )

        return sorted(fused, key=lambda x: x.score, reverse=True)[:limit]

    @staticmethod
    def _extract_query_terms(query_text: str) -> set[str]:
        """Extract salient query terms for lightweight lexical bonus."""
        tokens = re.findall(r"[a-zA-Zàèéìòù0-9\-]+", query_text.lower())
        stopwords = {
            "della",
            "delle",
            "degli",
            "dello",
            "dalla",
            "dalle",
            "dello",
            "dell",
            "degli",
            "dell'",
            "d'",
            "dell",
            "dati",
            "dopo",
            "come",
            "nella",
            "nelle",
            "nello",
            "della",
            "delle",
            "dello",
            "alla",
            "alle",
            "agli",
            "allo",
            "sono",
            "se",
            "che",
            "con",
            "per",
            "del",
            "dei",
            "di",
            "il",
            "lo",
            "la",
            "i",
            "gli",
            "le",
            "un",
            "una",
            "uno",
            "e",
            "ed",
            "o",
            "ai",
            "al",
            "nel",
            "nei",
            "in",
        }
        return {
            t
            for t in tokens
            if len(t) >= settings.search_hybrid_min_keyword_length
            and t not in stopwords
        }

    def _keyword_overlap_bonus(
        self, item: ArticleResult, query_terms: set[str]
    ) -> float:
        """Small lexical boost when title/article text matches salient query terms."""
        if not query_terms:
            return 0.0
        text = f"{item.articolo} {item.titolo}".lower()
        item_terms = set(re.findall(r"[a-zA-Zàèéìòù0-9\-]+", text))
        if not item_terms:
            return 0.0
        overlap = len(query_terms & item_terms) / len(query_terms)
        return min(
            settings.search_hybrid_keyword_bonus_max,
            overlap * settings.search_hybrid_keyword_bonus_scale,
        )

    def build_search_filters(
        self,
        classification: ClassificationResult,
        use_top_n_libri: int = settings.search_use_top_n_libri,
    ) -> list[tuple[str, str]]:
        """
        Convert classifier output to vector-search filters.

        For codici with no libri (e.g. amministrativo), `libro` is empty and
        vector search runs at source-level only.
        """
        selected = classification.libro_mappings[:use_top_n_libri]
        filters: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for source, libro in selected:
            source_norm = (source or "").strip()
            if not source_norm or source_norm == "unknown":
                continue
            libro_norm = (libro or "").strip()
            if source_norm == "codice_amministrativo":
                libro_norm = ""
            key = (source_norm, libro_norm)
            if key in seen:
                continue
            seen.add(key)
            filters.append(key)

        if not filters:
            return [
                ("codice_civile", ""),
                ("codice_penale", ""),
                ("codice_amministrativo", ""),
            ]
        return filters

    def search(
        self,
        claim: str,
        top_k: int = settings.search_top_k_default,
        use_top_n_libri: int = settings.search_use_top_n_libri,
    ) -> SearchResult:
        """
        Complete search pipeline for a legal claim.

        Args:
            claim: The legal claim text.
            top_k: Number of articles to return.
            use_top_n_libri: Number of top classified libri to search in.

        Returns:
            SearchResult with classification and relevant articles.
        """
        # Step 1: Classify the claim
        print(f"🔄 Classifying claim... (top_k={top_k}, libri_top_n={use_top_n_libri})")
        classification = self.classifier.classify(claim)
        print(f"   ✅ Classified into: {classification.categories}")

        # Step 2: Generate embedding
        print("🔄 Generating embedding...")
        embedding = self.embed_text(claim)
        print(f"   ✅ Embedding generated (dim: {len(embedding)})")

        # Step 3: Hybrid retrieval (vector + fulltext) filtered by libri/source.
        print("🔄 Searching in Neo4j (hybrid retrieval)...")
        libri_filters = self.build_search_filters(classification, use_top_n_libri)
        articles = self.vector_search(
            embedding,
            libri_filters,
            top_k,
            query_text=claim,
        )
        print(f"   ✅ Found {len(articles)} relevant articles")

        return SearchResult(
            claim=claim,
            classification=classification,
            articles=articles,
        )


def main():
    """Interactive CLI for testing the search pipeline."""
    print("=" * 70)
    print("LexCausa - Legal Search Pipeline")
    print("=" * 70)
    print()
    print("This tool searches Italian law codes for relevant articles.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    try:
        pipeline = LegalSearchPipeline()
        print()
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return

    while True:
        print("-" * 70)
        claim = input("📝 Enter your legal claim:\n> ").strip()

        if claim.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        if not claim:
            print("⚠️ Please enter a valid claim.")
            continue

        print()

        try:
            result = pipeline.search(
                claim,
                top_k=settings.search_top_k_default,
                use_top_n_libri=settings.search_use_top_n_libri,
            )
            print()
            print(result)
        except Exception as e:
            print(f"❌ Error during search: {e}")
            import traceback

            traceback.print_exc()

    pipeline.close()


if __name__ == "__main__":
    main()
