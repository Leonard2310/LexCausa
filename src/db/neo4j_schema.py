"""
Neo4j schema migrations - Vector Indexes for LexCausa.

Creates vector indexes for:
1. statutes_idx - Codice Penale + Codice Civile (articoli con embedding)
2. precedents_idx - Precedenti giurisprudenziali (itacasehold)

Crea anche:
- Nodi Libro per raggruppare gli articoli per libro di appartenenza
- Nodi Codice per raggruppare i libri per codice (Penale/Civile)
- Indice full-text per ricerca testuale

NOTA: Questo file è deprecato. Usa db_orchestrator.py per il setup.
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

# Embedding dimension - legal-bert-base-uncased = 768
EMBEDDING_DIM = 768

# Vector Index definitions
VECTOR_INDEXES = [
    {
        "name": "statutes_idx",
        "label": "Statute",
        "property": "embedding",
        "description": "Indice vettoriale per Codice Penale e Codice Civile",
    },
    {
        "name": "precedents_idx",
        "label": "Precedent",
        "property": "embedding",
        "description": "Indice vettoriale per precedenti (itacasehold)",
    },
]

# Fulltext index definitions for hybrid search
FULLTEXT_INDEXES = [
    {
        "name": "statutes_fulltext_idx",
        "labels": ["Statute"],
        "properties": ["titolo", "testo", "full_text"],
        "description": "Indice full-text per ricerca testuale negli statuti",
    },
]

# Constraint definitions for data integrity
CONSTRAINTS = [
    {
        "name": "statute_unique_id",
        "label": "Statute",
        "property": "statute_id",
        "description": "ID univoco per ogni articolo di statuto",
    },
    {
        "name": "libro_unique_name",
        "label": "Libro",
        "property": "name",
        "description": "Nome univoco per ogni libro",
    },
    {
        "name": "codice_unique_name",
        "label": "Codice",
        "property": "name",
        "description": "Nome univoco per ogni codice (Penale/Civile)",
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


def create_fulltext_index(session, index_config: dict) -> None:
    """Create a fulltext index if it doesn't exist."""
    labels = "|".join(index_config["labels"])
    properties = ", ".join([f"n.{p}" for p in index_config["properties"]])

    query = f"""
    CREATE FULLTEXT INDEX {index_config['name']} IF NOT EXISTS
    FOR (n:{labels})
    ON EACH [{properties}]
    """
    session.run(query)


def create_constraint(session, constraint_config: dict) -> None:
    """Create a uniqueness constraint if it doesn't exist."""
    query = f"""
    CREATE CONSTRAINT {constraint_config['name']} IF NOT EXISTS
    FOR (n:{constraint_config['label']})
    REQUIRE n.{constraint_config['property']} IS UNIQUE
    """
    session.run(query)


def create_graph_structure(session) -> None:
    """
    Create the hierarchical graph structure:

    (Codice) -[CONTAINS]-> (Libro) -[CONTAINS]-> (Statute)

    This enables queries like:
    - Get all articles from a specific libro
    - Navigate the legal code hierarchy
    - Group search results by libro/codice
    """
    # Create Codice nodes (Penale, Civile)
    session.run(
        """
        MERGE (c:Codice {name: 'Codice Penale'})
        SET c.description = 'Codice Penale Italiano'
    """
    )
    session.run(
        """
        MERGE (c:Codice {name: 'Codice Civile'})
        SET c.description = 'Codice Civile Italiano'
    """
    )

    # Libri del Codice Penale
    libri_penale = [
        ("Libro I", "Dei reati in generale"),
        ("Libro II", "Dei delitti in particolare"),
        ("Libro III", "Delle contravvenzioni in particolare"),
    ]

    for libro_name, libro_desc in libri_penale:
        session.run(
            """
            MERGE (l:Libro {name: $name, codice: 'codice_penale'})
            SET l.description = $description
            WITH l
            MATCH (c:Codice {name: 'Codice Penale'})
            MERGE (c)-[:CONTAINS]->(l)
        """,
            name=libro_name,
            description=libro_desc,
        )

    # Libri del Codice Civile
    libri_civile = [
        ("Libro Primo", "Delle persone e della famiglia"),
        ("Libro Secondo", "Delle successioni"),
        ("Libro Terzo", "Della proprietà"),
        ("Libro Quarto", "Delle obbligazioni"),
        ("Libro Quinto", "Del lavoro"),
        ("Libro Sesto", "Della tutela dei diritti"),
        ("Fuori range", "Articoli fuori dalla numerazione standard"),
    ]

    for libro_name, libro_desc in libri_civile:
        session.run(
            """
            MERGE (l:Libro {name: $name, codice: 'codice_civile'})
            SET l.description = $description
            WITH l
            MATCH (c:Codice {name: 'Codice Civile'})
            MERGE (c)-[:CONTAINS]->(l)
        """,
            name=libro_name,
            description=libro_desc,
        )

    print("✅ Graph structure (Codice -> Libro) created")


def main():
    """Create all indexes and schema for LexCausa."""
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))

    with driver.session() as session:
        # Create constraints first
        print("\n📋 Creating constraints...")
        for constraint_config in CONSTRAINTS:
            try:
                create_constraint(session, constraint_config)
                print(
                    f"   ✅ Constraint '{constraint_config['name']}' - "
                    f"{constraint_config['description']}"
                )
            except Exception as e:
                print(f"   ⚠️ Constraint '{constraint_config['name']}' skipped: {e}")

        # Create vector indexes
        print("\n🔍 Creating vector indexes...")
        for index_config in VECTOR_INDEXES:
            create_vector_index(session, index_config)
            print(
                f"   ✅ Vector index '{index_config['name']}' - "
                f"{index_config['description']}"
            )

        # Create fulltext indexes
        print("\n📝 Creating fulltext indexes...")
        for index_config in FULLTEXT_INDEXES:
            create_fulltext_index(session, index_config)
            print(
                f"   ✅ Fulltext index '{index_config['name']}' - "
                f"{index_config['description']}"
            )

        # Create graph structure
        print("\n🏗️ Creating graph structure...")
        create_graph_structure(session)

    driver.close()
    print("\n🎉 All schema migrations completed successfully!")


if __name__ == "__main__":
    main()
