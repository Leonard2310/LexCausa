#!/usr/bin/env python3
"""Verify precedents ingestion and vector search."""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    # Verifica stato indice
    result = session.run(
        "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties, options"
    )
    print("Vector indexes status:")
    for rec in result:
        if "VECTOR" in rec["type"]:
            dims = rec["options"].get("indexConfig", {}).get("vector.dimensions", "?")
            print(f"  {rec['name']}: {rec['state']} - {dims} dimensions")

    # Campione nodo precedent
    result = session.run(
        """
        MATCH (p:Precedent)
        RETURN p.title, p.materia, p.chunk_idx, size(p.embedding) as emb_size
        LIMIT 3
        """
    )
    print("\nSample Precedent nodes:")
    for rec in result:
        title = (rec["p.title"] or "")[:50]
        materia = rec["p.materia"]
        chunk = rec["p.chunk_idx"]
        emb_size = rec["emb_size"]
        print(f"  {title}... | {materia} | Chunk: {chunk} | Emb: {emb_size}")

    # Conta per materia
    result = session.run(
        """
        MATCH (p:Precedent)
        RETURN p.materia, count(*) as count
        ORDER BY count DESC
        """
    )
    print("\nPrecedents by materia:")
    for rec in result:
        print(f"  {rec['p.materia']}: {rec['count']} chunks")

driver.close()
print("\n✅ Verification complete!")
