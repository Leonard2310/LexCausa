"""
Neo4j schema migrations - Vector Indexes for LexCausa.

Creates 3 vector indexes:
1. statutes_idx - Codice Penale + Codice Civile
2. precedents_idx - Precedenti giurisprudenziali (itacasehold)
3. gazzetta_idx - Gazzetta Ufficiale
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

# Embedding dimension (es. OpenAI text-embedding-ada-002 = 1536)
EMBEDDING_DIM = 1536

# Index definitions
INDEXES = [
    {
        "name": "statutes_idx",
        "label": "Statute",
        "property": "embedding",
        "description": "Indice per Codice Penale e Codice Civile",
    },
    {
        "name": "precedents_idx",
        "label": "Precedent",
        "property": "embedding",
        "description": "Indice per precedenti giurisprudenziali (itacasehold)",
    },
    {
        "name": "normativa_idx",
        "label": "Normativa",
        "property": "embedding",
        "description": (
            "Indice per Gazzetta Ufficiale " "(Serie Generale + Corte Costituzionale)"
        ),
    },
]


def create_vector_index(session, index_config: dict) -> None:
    """Create a vector index if it doesn't exist."""
    query = f"""
    CREATE VECTOR INDEX {index_config['name']} IF NOT EXISTS
    FOR (n:{index_config['label']})
    ON (n.{index_config['property']})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    session.run(query)


def main():
    """Create all vector indexes for LexCausa."""
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))

    with driver.session() as session:
        for index_config in INDEXES:
            create_vector_index(session, index_config)
            print(
                f"✅ Index '{index_config['name']}' created - "
                f"{index_config['description']}"
            )

    driver.close()
    print("\n🎉 All vector indexes created successfully!")


if __name__ == "__main__":
    main()
