#!/usr/bin/env python3
"""Ingest itacasehold precedents with embeddings into Neo4j."""

import os
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_itacasehold_with_embeddings  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

# Load environment
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

BATCH_SIZE = 100


def main():
    print("=" * 60)
    print("Itacasehold Ingestion with Embeddings")
    print("=" * 60)

    # Load data
    print("\n1. Loading itacasehold data with embeddings...")
    metadata, embeddings = load_itacasehold_with_embeddings()

    # Connect to Neo4j
    print("\n2. Connecting to Neo4j...")
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))

    with driver.session() as session:
        # Clear existing Precedent nodes
        print("\n3. Clearing existing Precedent nodes...")
        result = session.run("MATCH (p:Precedent) RETURN count(p) as count")
        count = result.single()["count"]
        print(f"   Found {count} existing Precedent nodes")

        if count > 0:
            session.run("MATCH (p:Precedent) DETACH DELETE p")
            print(f"   Deleted {count} Precedent nodes")

        # Ingest new precedents with embeddings
        print(f"\n4. Ingesting {len(metadata)} precedent chunks...")

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
            if (i // BATCH_SIZE + 1) % 10 == 0 or inserted == len(metadata):
                print(f"   Inserted {inserted}/{len(metadata)} chunks")

        # Verify
        result = session.run("MATCH (p:Precedent) RETURN count(p) as count")
        final_count = result.single()["count"]
        print(f"\n✅ Ingestion complete: {final_count} Precedent nodes created")

        # Test vector index
        print("\n5. Testing vector search...")
        result = session.run(
            """
            MATCH (p:Precedent)
            WHERE p.embedding IS NOT NULL
            RETURN count(p) as count
        """
        )
        with_emb = result.single()["count"]
        print(f"   Precedent nodes with embeddings: {with_emb}")

    driver.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
