"""
Neo4j Ingestion - Loads data into Neo4j as nodes with embeddings.

This script creates nodes for:
1. Statute (Codice Penale + Codice Civile) -> statutes_idx
   - Includes libro di appartenenza and embeddings
   - Creates BELONGS_TO relationships with Libro nodes
2. Precedent (itacasehold) -> precedents_idx
3. Normativa -> normativa_idx

Vector search capabilities:
- Each Statute node contains an embedding property for vector similarity search
- Similar to OpenSearch/Elasticsearch but with graph traversal capabilities
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from data_loader import (
    load_codice_civile_with_embeddings,
    load_codice_penale_with_embeddings,
    load_gazzetta_ufficiale,
    load_precedents,
)
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

BATCH_SIZE = 100  # Reduced for embedding data


class Neo4jIngestion:
    """Handles data ingestion into Neo4j."""

    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))

    def close(self):
        self.driver.close()

    def clear_all_nodes(self, label: Optional[str] = None):
        """Delete all nodes (or nodes with a specific label)."""
        with self.driver.session() as session:
            if label:
                query = f"MATCH (n:{label}) DETACH DELETE n"
                session.run(query)
                print(f"🗑️ Deleted all {label} nodes")
            else:
                session.run("MATCH (n) DETACH DELETE n")
                print("🗑️ Deleted all nodes")

    def ingest_statutes(
        self, df: pd.DataFrame, source: str, embeddings: Optional[np.ndarray] = None
    ):
        """
        Ingest statute articles into Neo4j with embeddings and libro relationships.

        Args:
            df: DataFrame with columns articolo, titolo, testo, libro, source
            source: 'codice_penale' or 'codice_civile'
            embeddings: Optional numpy array of shape (n_articles, embedding_dim)

        Creates:
            - Statute nodes with embedding property
            - BELONGS_TO relationship with Libro nodes
        """
        print(f"📤 Ingesting {len(df)} {source} articles...")

        if embeddings is not None:
            if len(embeddings) != len(df):
                print(
                    f"⚠️ Embeddings count ({len(embeddings)}) "
                    f"!= articles count ({len(df)})"
                )
                print("   Embeddings will not be added")
                embeddings = None
            else:
                print(f"   📊 Embeddings shape: {embeddings.shape}")

        with self.driver.session() as session:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                batch_embeddings = None
                if embeddings is not None:
                    batch_embeddings = embeddings[i : i + BATCH_SIZE]

                records = batch.to_dict("records")

                # Clean records for Neo4j
                clean_records = []
                for idx, record in enumerate(records):
                    # Global index for unique ID
                    global_idx = i + idx

                    # Normalize article identifier
                    if source == "codice_civile":
                        articolo = str(record.get("article_id", ""))
                        titolo = str(record.get("article_title", ""))
                        testo = str(record.get("article_text", ""))
                    else:  # codice_penale
                        articolo = str(record.get("articolo", ""))
                        titolo = str(record.get("titolo", ""))
                        testo = str(record.get("testo", ""))

                    libro = (
                        str(record.get("libro", ""))
                        or str(record.get("libro_codice_penale", ""))
                        or str(record.get("libro_codice_civile", ""))
                    )

                    # Use global index to ensure unique ID
                    # (handles duplicates like art1159)
                    clean_record = {
                        "statute_id": f"{source}_{global_idx}_{articolo}",
                        "articolo": articolo,
                        "titolo": titolo,
                        "testo": testo,
                        "libro": libro,
                        "source": source,
                        "full_text": f"Art. {articolo} - {titolo}: {testo}",
                    }

                    # Add embedding if available
                    if batch_embeddings is not None:
                        clean_record["embedding"] = batch_embeddings[idx].tolist()

                    clean_records.append(clean_record)

                # Create Statute nodes with embeddings
                if embeddings is not None:
                    session.run(
                        """
                        UNWIND $records AS record
                        CREATE (s:Statute {
                            statute_id: record.statute_id,
                            articolo: record.articolo,
                            titolo: record.titolo,
                            testo: record.testo,
                            libro: record.libro,
                            source: record.source,
                            full_text: record.full_text,
                            embedding: record.embedding
                        })
                        WITH s, record
                        MATCH (l:Libro {name: record.libro, codice: record.source})
                        MERGE (s)-[:BELONGS_TO]->(l)
                        """,
                        records=clean_records,
                    )
                else:
                    # Without embeddings
                    session.run(
                        """
                        UNWIND $records AS record
                        CREATE (s:Statute {
                            statute_id: record.statute_id,
                            articolo: record.articolo,
                            titolo: record.titolo,
                            testo: record.testo,
                            libro: record.libro,
                            source: record.source,
                            full_text: record.full_text
                        })
                        WITH s, record
                        MATCH (l:Libro {name: record.libro, codice: record.source})
                        MERGE (s)-[:BELONGS_TO]->(l)
                        """,
                        records=clean_records,
                    )

                inserted = min(i + BATCH_SIZE, len(df))
                print(
                    f"   Inserted batch {i // BATCH_SIZE + 1} "
                    f"({inserted}/{len(df)})"
                )

        print(
            f"✅ Ingested {len(df)} {source} articles"
            + (" with embeddings" if embeddings is not None else "")
        )

    def ingest_precedents(self, df: pd.DataFrame):
        """
        Ingest precedents into Neo4j.

        Expected columns from itacasehold: context, question, holdings, etc.
        """
        print(f"📤 Ingesting {len(df)} precedents...")

        with self.driver.session() as session:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                records = batch.to_dict("records")

                # Clean records for Neo4j
                clean_records = []
                for idx, record in enumerate(records):
                    clean_record = {
                        "id": f"prec_{i + idx}",
                        # Limit size
                        "context": str(record.get("context", ""))[:50000],
                        "question": str(record.get("question", "")),
                        "label": (
                            int(record.get("label", 0))
                            if pd.notna(record.get("label"))
                            else 0
                        ),
                        "source": str(record.get("source", "itacasehold")),
                    }
                    # Add holdings if present
                    # itacasehold has holding_0 to holding_4
                    for j in range(5):
                        holding_key = f"holding_{j}"
                        if holding_key in record:
                            clean_record[holding_key] = str(record.get(holding_key, ""))

                    # Create full text for search
                    clean_record["full_text"] = (
                        f"{clean_record['context']} {clean_record['question']}"
                    )
                    clean_records.append(clean_record)

                session.run(
                    """
                    UNWIND $records AS record
                    CREATE (p:Precedent {
                        id: record.id,
                        context: record.context,
                        question: record.question,
                        label: record.label,
                        source: record.source,
                        full_text: record.full_text
                    })
                    """,
                    records=clean_records,
                )

                inserted = min(i + BATCH_SIZE, len(df))
                print(
                    f"   Inserted batch {i // BATCH_SIZE + 1} "
                    f"({inserted}/{len(df)})"
                )

        print(f"✅ Ingested {len(df)} precedents")

    def ingest_precedents_with_embeddings(
        self, metadata: List[dict], embeddings: np.ndarray
    ):
        """
        Ingest precedent chunks into Neo4j with embeddings.

        Each chunk becomes a PrecedentChunk node with:
        - chunk_id: unique identifier
        - doc_id: reference to original document
        - chunk_idx: index within the document
        - title, summary, materia, url: from original document
        - chunk_text: first 500 chars of chunk for preview
        - embedding: vector for similarity search

        Args:
            metadata: List of dicts with chunk metadata
            embeddings: numpy array of shape (n_chunks, 768)
        """
        print(f"📤 Ingesting {len(metadata)} precedent chunks with embeddings...")

        if len(metadata) != embeddings.shape[0]:
            print(
                f"⚠️ Mismatch: {len(metadata)} metadata "
                f"vs {embeddings.shape[0]} embeddings"
            )
            return

        with self.driver.session() as session:
            for i in range(0, len(metadata), BATCH_SIZE):
                batch_meta = metadata[i : i + BATCH_SIZE]
                batch_emb = embeddings[i : i + BATCH_SIZE]

                records = []
                for idx, (meta, emb) in enumerate(zip(batch_meta, batch_emb)):
                    global_idx = i + idx
                    records.append(
                        {
                            "chunk_id": f"prec_chunk_{global_idx}",
                            "doc_id": meta.get("doc_id", 0),
                            "chunk_idx": meta.get("chunk_idx", 0),
                            "title": str(meta.get("title", ""))[:500],
                            "summary": str(meta.get("summary", ""))[:2000],
                            "materia": str(meta.get("materia", "")),
                            "url": str(meta.get("url", "")),
                            "chunk_text": str(meta.get("chunk_text", ""))[:500],
                            "embedding": emb.tolist(),
                            "source": "itacasehold",
                        }
                    )

                session.run(
                    """
                    UNWIND $records AS record
                    CREATE (p:Precedent {
                        chunk_id: record.chunk_id,
                        doc_id: record.doc_id,
                        chunk_idx: record.chunk_idx,
                        title: record.title,
                        summary: record.summary,
                        materia: record.materia,
                        url: record.url,
                        chunk_text: record.chunk_text,
                        embedding: record.embedding,
                        source: record.source
                    })
                    """,
                    records=records,
                )

                inserted = min(i + BATCH_SIZE, len(metadata))
                print(
                    f"   Inserted batch {i // BATCH_SIZE + 1} "
                    f"({inserted}/{len(metadata)})"
                )

        print(f"✅ Ingested {len(metadata)} precedent chunks with embeddings")

    def ingest_normativa(self, df: pd.DataFrame):
        """
        Ingest Gazzetta Ufficiale into Neo4j.

        Includes Serie Generale + Corte Costituzionale.
        """
        print(f"📤 Ingesting {len(df)} Normativa documents...")

        with self.driver.session() as session:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                records = batch.to_dict("records")

                # Clean records for Neo4j
                clean_records = []
                for idx, record in enumerate(records):
                    clean_record = {
                        "id": f"norm_{i + idx}",
                        "type": str(record.get("type", "")),
                        "year": str(record.get("year", "")),
                        "title": str(record.get("title", "")),
                        "subtitle": str(record.get("subtitle", "")),
                        "rubrica": str(record.get("rubrica", "")),
                        "emettitore": str(record.get("emettitore", "")),
                        "intestazione": str(record.get("intestazione", "")),
                        # Limit size
                        "text": str(record.get("text", ""))[:100000],
                        "url": str(record.get("url", "")),
                        "source": "gazzetta_ufficiale",
                    }
                    # Create full text for embedding/search
                    clean_record["full_text"] = (
                        f"{clean_record['title']} - "
                        f"{clean_record['subtitle']} "
                        f"{clean_record['rubrica']} "
                        f"{clean_record['text'][:10000]}"
                    )
                    clean_records.append(clean_record)

                session.run(
                    """
                    UNWIND $records AS record
                    CREATE (n:Normativa {
                        id: record.id,
                        type: record.type,
                        year: record.year,
                        title: record.title,
                        subtitle: record.subtitle,
                        rubrica: record.rubrica,
                        emettitore: record.emettitore,
                        intestazione: record.intestazione,
                        text: record.text,
                        url: record.url,
                        source: record.source,
                        full_text: record.full_text
                    })
                    """,
                    records=clean_records,
                )

                inserted = min(i + BATCH_SIZE, len(df))
                print(
                    f"   Inserted batch {i // BATCH_SIZE + 1} "
                    f"({inserted}/{len(df)})"
                )

        print(f"✅ Ingested {len(df)} Normativa documents")

    def get_node_counts(self) -> dict[str, int]:
        """Get count of nodes for each label."""
        counts = {}
        with self.driver.session() as session:
            for label in ["Statute", "Precedent", "Normativa", "Libro", "Codice"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                counts[label] = result.single()["count"]
        return counts

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        source: Optional[str] = None,
        libro: Optional[str] = None,
    ) -> List[dict]:
        """
        Perform vector similarity search on statutes.

        Similar to OpenSearch/Elasticsearch kNN search but with graph capabilities.

        Args:
            query_embedding: The embedding vector of the query (768 dimensions)
            top_k: Number of results to return
            source: Optional filter by source ('codice_penale' or 'codice_civile')
            libro: Optional filter by libro name

        Returns:
            List of matching statutes with similarity scores
        """
        with self.driver.session() as session:
            # Build filter conditions
            filters = []
            if source:
                filters.append(f"node.source = '{source}'")
            if libro:
                filters.append(f"node.libro = '{libro}'")

            where_clause = ""
            if filters:
                where_clause = "WHERE " + " AND ".join(filters)

            query = f"""
            CALL db.index.vector.queryNodes('statutes_idx', $top_k, $embedding)
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
            LIMIT $top_k
            """

            result = session.run(query, embedding=query_embedding, top_k=top_k)
            return [dict(record) for record in result]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> List[dict]:
        """
        Perform hybrid search combining vector similarity and fulltext search.

        Args:
            query_text: Text query for fulltext search
            query_embedding: Embedding vector for semantic search
            top_k: Number of results to return
            vector_weight: Weight for vector search (0-1), fulltext = 1 - vector_weight

        Returns:
            List of matching statutes with combined scores
        """
        with self.driver.session() as session:
            query = """
            // Vector search
            CALL db.index.vector.queryNodes(
                'statutes_idx', $top_k * 2, $embedding
            )
            YIELD node AS vNode, score AS vScore
            WITH collect({node: vNode, score: vScore}) AS vectorResults

            // Fulltext search
            CALL db.index.fulltext.queryNodes(
                'statutes_fulltext_idx', $query_text
            )
            YIELD node AS fNode, score AS fScore
            WITH vectorResults,
                 collect({node: fNode, score: fScore}) AS fulltextResults

            // Combine results
            UNWIND vectorResults AS vr
            WITH vr.node AS node, vr.score AS vScore, fulltextResults
            OPTIONAL MATCH (fResult)
                WHERE fResult IN [f IN fulltextResults | f.node]
                AND fResult = node
            WITH node, vScore,
                 COALESCE(
                     [f IN fulltextResults WHERE f.node = node | f.score][0],
                     0
                 ) AS fScore

            // Calculate hybrid score
            WITH node,
                 (vScore * $vector_weight + fScore * (1 - $vector_weight))
                 AS hybridScore
            RETURN node.statute_id AS id,
                   node.articolo AS articolo,
                   node.titolo AS titolo,
                   node.testo AS testo,
                   node.libro AS libro,
                   node.source AS source,
                   hybridScore AS score
            ORDER BY hybridScore DESC
            LIMIT $top_k
            """

            result = session.run(
                query,
                embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                vector_weight=vector_weight,
            )
            return [dict(record) for record in result]

    def get_articles_by_libro(self, libro: str, codice: str) -> List[dict]:
        """
        Get all articles belonging to a specific libro.

        Args:
            libro: Name of the libro (e.g., 'Libro I', 'Libro Primo')
            codice: 'codice_penale' or 'codice_civile'

        Returns:
            List of articles in the libro
        """
        with self.driver.session() as session:
            query = """
            MATCH (s:Statute)-[:BELONGS_TO]->(l:Libro {name: $libro, codice: $codice})
            RETURN s.statute_id AS id,
                   s.articolo AS articolo,
                   s.titolo AS titolo,
                   s.testo AS testo
            ORDER BY s.articolo
            """
            result = session.run(query, libro=libro, codice=codice)
            return [dict(record) for record in result]

    def find_similar_in_libro(
        self, query_embedding: List[float], libro: str, top_k: int = 5
    ) -> List[dict]:
        """
        Find similar articles within a specific libro.

        Combines vector search with graph traversal.

        Args:
            query_embedding: The embedding vector of the query
            libro: Name of the libro to search within
            top_k: Number of results

        Returns:
            List of similar articles from the specified libro
        """
        with self.driver.session() as session:
            query = """
            CALL db.index.vector.queryNodes('statutes_idx', $top_k * 3, $embedding)
            YIELD node, score
            MATCH (node)-[:BELONGS_TO]->(l:Libro {name: $libro})
            RETURN node.statute_id AS id,
                   node.articolo AS articolo,
                   node.titolo AS titolo,
                   node.testo AS testo,
                   node.source AS source,
                   l.name AS libro,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """
            result = session.run(
                query, embedding=query_embedding, libro=libro, top_k=top_k
            )
            return [dict(record) for record in result]


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("LexCausa - Neo4j Data Ingestion")
    print("=" * 60)

    ingestion = Neo4jIngestion()

    try:
        # Clear existing nodes (optional - comment out to append)
        print("\n🗑️ Clearing existing nodes...")
        ingestion.clear_all_nodes()

        # Run schema migrations first (creates Libro nodes)
        print("\n🏗️ Running schema migrations...")
        from neo4j_schema import main as run_schema

        run_schema()

        # Ingest Statutes with embeddings
        print("\n📖 Processing Codice Penale...")
        codice_penale, penale_embeddings = load_codice_penale_with_embeddings()
        ingestion.ingest_statutes(codice_penale, "codice_penale", penale_embeddings)

        print("\n📖 Processing Codice Civile...")
        codice_civile, civile_embeddings = load_codice_civile_with_embeddings()
        ingestion.ingest_statutes(codice_civile, "codice_civile", civile_embeddings)

        # Ingest Precedents
        print("\n⚖️ Processing Precedenti...")
        precedents = load_precedents("train")  # Start with train split
        ingestion.ingest_precedents(precedents)

        # Ingest Normativa (Serie Generale + Corte Costituzionale)
        print("\n📰 Processing Normativa " "(Serie Generale + Corte Costituzionale)...")
        gazzetta = load_gazzetta_ufficiale()  # Full dataset
        ingestion.ingest_normativa(gazzetta)

        # Summary
        print("\n" + "=" * 60)
        print("📊 Ingestion Summary:")
        counts = ingestion.get_node_counts()
        for label, count in counts.items():
            print(f"   {label}: {count} nodes")
        print("=" * 60)
        print("✅ Ingestion complete!")
        print("\n🔍 Vector search ready! Use vector_search() to find similar articles.")

    finally:
        ingestion.close()


if __name__ == "__main__":
    main()
