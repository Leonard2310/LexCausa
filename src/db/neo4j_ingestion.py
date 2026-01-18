"""
Neo4j Ingestion - Loads data into Neo4j as nodes.

This script creates nodes for:
1. Statute (Codice Penale + Codice Civile) -> statutes_idx
2. Precedent (itacasehold) -> precedents_idx
3. GazzettaUfficiale -> gazzetta_idx

Note: Embeddings must be generated separately and added to nodes
before vector search can be performed.
"""

import os
from typing import Optional

import pandas as pd
from data_loader import (
    load_codice_civile,
    load_codice_penale,
    load_gazzetta_ufficiale,
    load_precedents,
)
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

BATCH_SIZE = 500  # Number of records to insert per transaction


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

    def ingest_statutes(self, df: pd.DataFrame, source: str):
        """
        Ingest statute articles into Neo4j.

        Expected columns: articolo, titolo, testo, reference,
        external_reference, source
        """
        print(f"📤 Ingesting {len(df)} {source} articles...")

        with self.driver.session() as session:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                records = batch.to_dict("records")

                # Clean records for Neo4j
                clean_records = []
                for record in records:
                    clean_record = {
                        "articolo": str(record.get("articolo", "")),
                        "titolo": str(record.get("titolo", "")),
                        "testo": str(record.get("testo", "")),
                        "source": source,
                        # Combine for full text search
                        "full_text": (
                            f"Art. {record.get('articolo', '')} - "
                            f"{record.get('titolo', '')}: "
                            f"{record.get('testo', '')}"
                        ),
                    }
                    clean_records.append(clean_record)

                session.run(
                    """
                    UNWIND $records AS record
                    CREATE (s:Statute {
                        articolo: record.articolo,
                        titolo: record.titolo,
                        testo: record.testo,
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

        print(f"✅ Ingested {len(df)} {source} articles")

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
            for label in ["Statute", "Precedent", "Normativa"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                counts[label] = result.single()["count"]
        return counts


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

        # Ingest Statutes
        print("\n📖 Processing Codice Penale...")
        codice_penale = load_codice_penale()
        ingestion.ingest_statutes(codice_penale, "codice_penale")

        print("\n📖 Processing Codice Civile...")
        codice_civile = load_codice_civile()
        ingestion.ingest_statutes(codice_civile, "codice_civile")

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

    finally:
        ingestion.close()


if __name__ == "__main__":
    main()
