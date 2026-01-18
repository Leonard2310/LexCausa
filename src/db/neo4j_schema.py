"""
Neo4j schema migrations - Vector Index for document embeddings.
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

INDEX_NAME = "doc_embedding_idx"
DIM = 1536  # Dimensione degli embeddings (es. OpenAI text-embedding-ada-002)

CREATE_INDEX = f"""
CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
FOR (d:Document)
ON (d.embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {DIM},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""


def main():
    """Create vector index for document embeddings."""
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    with driver.session() as session:
        session.run(CREATE_INDEX)
        print(f"✅ Vector index '{INDEX_NAME}' created successfully!")
    driver.close()


if __name__ == "__main__":
    main()
