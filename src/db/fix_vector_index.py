#!/usr/bin/env python3
"""Recreate Neo4j vector indexes with correct 768 dimensions."""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
env_path = os.path.join(project_root, ".env")
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
pwd = os.getenv("NEO4J_PASSWORD")
print(f"Connecting to: {uri}")

driver = GraphDatabase.driver(uri, auth=(user, pwd))

# Indexes to fix
INDEXES_TO_FIX = [
    ("statutes_idx", "Statute"),
    ("precedents_idx", "Precedent"),
]

with driver.session() as session:
    for index_name, label in INDEXES_TO_FIX:
        print(f"\nFixing {index_name}...")

        # Drop existing index
        print(f"  Dropping {index_name}...")
        session.run(f"DROP INDEX {index_name} IF EXISTS")

        # Recreate with 768 dimensions
        print(f"  Creating {index_name} with 768 dimensions...")
        session.run(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'COSINE',
                    `vector.quantization.enabled`: true
                }}
            }}
        """
        )

        # Verify
        result = session.run(
            f"""
            SHOW INDEXES YIELD name, options
            WHERE name = "{index_name}"
            RETURN options
        """
        )
        for record in result:
            dim = record["options"]["indexConfig"]["vector.dimensions"]
            print(f"  ✅ {index_name} recreated with {dim} dimensions")

print("\nDone!")
driver.close()
