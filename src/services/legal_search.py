"""
Legal Search Pipeline for LexCausa.

Complete pipeline that:
1. Classifies a legal claim using Groq Cloud LLM
2. Generates embedding for the claim using Legal-BERT
3. Performs vector search in Neo4j filtered by relevant libri/source
4. Returns the most relevant articles from Italian law codes
"""

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
    ) -> list[ArticleResult]:
        """
        Perform vector search filtered by libri.

        Args:
            embedding: Query embedding vector.
            libri_filters: List of (source, libro) tuples to filter by.
            top_k: Number of results per libro.

        Returns:
            List of ArticleResult sorted by score.
        """
        all_results = []

        with self.driver.session() as session:
            for source, libro in libri_filters:
                if not source or source == "unknown":
                    continue

                # For codici without libri (e.g. amministrativo), search by source only.
                if libro:
                    query = """
                    CALL db.index.vector.queryNodes(
                        'statutes_idx', $top_k_expanded, $embedding
                    )
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
                    LIMIT $top_k
                    """
                    result = session.run(
                        query,
                        embedding=embedding,
                        source=source,
                        libro=libro,
                        top_k=top_k,
                        top_k_expanded=top_k * 10,
                    )
                else:
                    query = """
                    CALL db.index.vector.queryNodes(
                        'statutes_idx', $top_k_expanded, $embedding
                    )
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
                    LIMIT $top_k
                    """
                    result = session.run(
                        query,
                        embedding=embedding,
                        source=source,
                        top_k=top_k,
                        top_k_expanded=top_k * 20,
                    )

                for record in result:
                    all_results.append(
                        ArticleResult(
                            statute_id=record["id"],
                            articolo=record["articolo"],
                            titolo=record["titolo"],
                            testo=record["testo"] or "",
                            libro=record["libro"],
                            source=record["source"],
                            score=record["score"],
                        )
                    )

        # Sort by score and deduplicate
        seen_ids = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.statute_id not in seen_ids:
                seen_ids.add(r.statute_id)
                unique_results.append(r)

        return unique_results[:top_k]

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
            return [("codice_civile", ""), ("codice_penale", "")]
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

        # Step 3: Vector search filtered by libri/source
        print("🔄 Searching in Neo4j...")
        libri_filters = self.build_search_filters(classification, use_top_n_libri)
        articles = self.vector_search(embedding, libri_filters, top_k)
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
