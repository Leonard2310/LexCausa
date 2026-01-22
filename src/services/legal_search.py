"""
Legal Search Pipeline for LexCausa.

Complete pipeline that:
1. Classifies a legal claim using Groq Cloud LLM
2. Generates embedding for the claim using Legal-BERT
3. Performs vector search in Neo4j filtered by relevant libri
4. Returns the most relevant articles from Italian law codes
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from transformers import AutoModel, AutoTokenizer

from .claim_classifier import ClaimClassifier, ClassificationResult

load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Embedding model configuration (same as used for indexing)
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512


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
        source_label = (
            "Codice Civile" if self.source == "codice_civile" else "Codice Penale"
        )
        return (
            f"[{source_label}] Art. {self.articolo} - {self.titolo}\n"
            f"  Libro: {self.libro}\n"
            f"  Score: {self.score:.4f}\n"
            f"  Testo: {self.testo[:200]}..."
        )


@dataclass
class SearchResult:
    """Complete search result with classification and articles."""

    claim: str
    classification: ClassificationResult
    articles: list[ArticleResult] = field(default_factory=list)
    embedding_model: str = MODEL_NAME

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
            groq_api_key: API key for Groq Cloud. If None, reads from env.
            device: Device for embeddings ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Initialize classifier
        self.classifier = ClaimClassifier(api_key=groq_api_key)

        # Initialize embedding model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading embedding model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
        # Use local cached model (pre-downloaded) with weights_only=False
        # to avoid network calls and torch.load security check
        import os

        os.environ["TRUST_REMOTE_CODE"] = "1"
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, local_files_only=True, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"✅ Embedding model loaded: {MODEL_NAME}")

        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
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
            max_length=MAX_LENGTH,
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
        top_k: int = 10,
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
                # Build query with filters
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
                    top_k_expanded=top_k * 10,  # Expand for filtering
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

    def search(
        self,
        claim: str,
        top_k: int = 10,
        use_top_n_libri: int = 3,
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
        print("🔄 Classifying claim...")
        classification = self.classifier.classify(claim)
        print(f"   ✅ Classified into: {classification.categories}")

        # Step 2: Generate embedding
        print("🔄 Generating embedding...")
        embedding = self.embed_text(claim)
        print(f"   ✅ Embedding generated (dim: {len(embedding)})")

        # Step 3: Vector search filtered by libri
        print("🔄 Searching in Neo4j...")
        libri_filters = classification.libro_mappings[:use_top_n_libri]
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
            result = pipeline.search(claim, top_k=5)
            print()
            print(result)
        except Exception as e:
            print(f"❌ Error during search: {e}")
            import traceback

            traceback.print_exc()

    pipeline.close()


if __name__ == "__main__":
    main()
