#!/usr/bin/env python3
"""
LexCausa Database Orchestrator.

Esegue in ordine tutte le operazioni necessarie per inizializzare il database:
1. Verifica connessione Neo4j
2. Pulizia database (opzionale)
3. Creazione schema (indici, constraint, struttura grafo)
4. Caricamento statuti (Codice Penale + Civile + Amministrativo) con embeddings
5. Caricamento precedenti (itacasehold) senza embeddings

Uso:
    python db_orchestrator.py                    # Setup completo
    python db_orchestrator.py --clean            # Pulisce e reinizializza tutto
    python db_orchestrator.py --check            # Solo verifica stato DB
    python db_orchestrator.py --statutes         # Ricarica solo statuti
    python db_orchestrator.py --precedents       # Ricarica solo precedenti
    python db_orchestrator.py --clean --statutes # Pulisce e carica solo statuti
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # src/db
sys.path.insert(0, str(Path(__file__).parent.parent))  # src

from dotenv import load_dotenv  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "password")

EMBEDDING_DIM = 768
BATCH_SIZE = 100


class DatabaseOrchestrator:
    """Gestisce l'inizializzazione completa del database Neo4j."""

    def __init__(self):
        self.driver: Optional[GraphDatabase.driver] = None
        self._connect()

    def _connect(self) -> bool:
        """Connessione a Neo4j."""
        try:
            self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))
            self.driver.verify_connectivity()
            print(f"✅ Connesso a Neo4j: {URI}")
            return True
        except Exception as e:
            print(f"❌ Errore connessione Neo4j: {e}")
            return False

    def close(self):
        """Chiude la connessione."""
        if self.driver:
            self.driver.close()

    # =========================================================================
    # STEP 1: VERIFICA E PULIZIA
    # =========================================================================

    def check_status(self) -> dict:
        """Verifica lo stato attuale del database."""
        status = {
            "connected": False,
            "nodes": {},
            "indexes": [],
            "constraints": [],
        }

        if not self.driver:
            return status

        status["connected"] = True

        with self.driver.session() as session:
            # Conta nodi per label
            for label in ["Statute", "Precedent", "Libro", "Codice"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                status["nodes"][label] = result.single()["count"]

            # Lista indici
            result = session.run("SHOW INDEXES YIELD name, type, state")
            for record in result:
                status["indexes"].append(
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "state": record["state"],
                    }
                )

            # Lista constraint
            result = session.run("SHOW CONSTRAINTS YIELD name, type")
            for record in result:
                status["constraints"].append(
                    {
                        "name": record["name"],
                        "type": record["type"],
                    }
                )

        return status

    def print_status(self):
        """Stampa lo stato del database."""
        status = self.check_status()

        print("\n" + "=" * 60)
        print("📊 STATO DATABASE")
        print("=" * 60)

        if not status["connected"]:
            print("❌ Non connesso a Neo4j")
            return

        print("\n📦 Nodi:")
        for label, count in status["nodes"].items():
            emoji = "✅" if count > 0 else "⚪"
            print(f"   {emoji} {label}: {count}")

        print("\n🔍 Indici:")
        for idx in status["indexes"]:
            state_emoji = "✅" if idx["state"] == "ONLINE" else "⏳"
            print(f"   {state_emoji} {idx['name']} ({idx['type']}) - {idx['state']}")

        if not status["indexes"]:
            print("   ⚪ Nessun indice")

        print("\n🔒 Constraint:")
        for c in status["constraints"]:
            print(f"   ✅ {c['name']} ({c['type']})")

        if not status["constraints"]:
            print("   ⚪ Nessun constraint")

        print("=" * 60)

    def clean_database(self):
        """Pulisce completamente il database."""
        print("\n🗑️ Pulizia database...")

        with self.driver.session() as session:
            # Drop tutti gli indici
            result = session.run("SHOW INDEXES YIELD name")
            for record in result:
                idx_name = record["name"]
                try:
                    session.run(f"DROP INDEX {idx_name} IF EXISTS")
                    print(f"   Dropped index: {idx_name}")
                except Exception:
                    pass

            # Drop tutti i constraint
            result = session.run("SHOW CONSTRAINTS YIELD name")
            for record in result:
                const_name = record["name"]
                try:
                    session.run(f"DROP CONSTRAINT {const_name} IF EXISTS")
                    print(f"   Dropped constraint: {const_name}")
                except Exception:
                    pass

            # Elimina tutti i nodi
            session.run("MATCH (n) DETACH DELETE n")
            print("   Eliminati tutti i nodi")

        print("✅ Database pulito")

    # =========================================================================
    # STEP 2: CREAZIONE SCHEMA
    # =========================================================================

    def create_schema(self):
        """Crea indici, constraint e struttura del grafo."""
        print("\n🏗️ Creazione schema...")

        with self.driver.session() as session:
            # -----------------------------------------------------------------
            # CONSTRAINT
            # -----------------------------------------------------------------
            print("\n   📋 Creazione constraint...")

            constraints = [
                ("statute_unique_id", "Statute", "statute_id"),
                ("codice_unique_name", "Codice", "name"),
                ("precedent_unique_id", "Precedent", "precedent_id"),
            ]

            for name, label, prop in constraints:
                try:
                    session.run(
                        f"""
                        CREATE CONSTRAINT {name} IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.{prop} IS UNIQUE
                    """
                    )
                    print(f"      ✅ {name}")
                except Exception as e:
                    print(f"      ⚠️ {name}: {e}")

            # Constraint composto per Libro (name + codice)
            try:
                session.run(
                    """
                    CREATE CONSTRAINT libro_unique_name_codice IF NOT EXISTS
                    FOR (n:Libro)
                    REQUIRE (n.name, n.codice) IS UNIQUE
                """
                )
                print("      ✅ libro_unique_name_codice")
            except Exception as e:
                print(f"      ⚠️ libro_unique_name_codice: {e}")

            # -----------------------------------------------------------------
            # VECTOR INDEXES
            # -----------------------------------------------------------------
            print("\n   🔍 Creazione indici vettoriali...")

            vector_indexes = [
                ("statutes_idx", "Statute", "embedding"),
            ]

            # Ensure deprecated precedent vector index is removed.
            session.run("DROP INDEX precedents_idx IF EXISTS")
            print("      ✅ precedents_idx rimosso (precedenti senza embedding)")

            for name, label, prop in vector_indexes:
                # Drop e ricrea per assicurare dimensione corretta
                session.run(f"DROP INDEX {name} IF EXISTS")
                session.run(
                    f"""
                    CREATE VECTOR INDEX {name} IF NOT EXISTS
                    FOR (n:{label}) ON (n.{prop})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {EMBEDDING_DIM},
                            `vector.similarity_function`: 'COSINE',
                            `vector.quantization.enabled`: true
                        }}
                    }}
                """
                )
                print(f"      ✅ {name} ({EMBEDDING_DIM} dim)")

            # -----------------------------------------------------------------
            # FULLTEXT INDEX
            # -----------------------------------------------------------------
            print("\n   📝 Creazione indici fulltext...")

            session.run(
                """
                CREATE FULLTEXT INDEX statutes_fulltext_idx IF NOT EXISTS
                FOR (n:Statute)
                ON EACH [n.titolo, n.testo, n.full_text]
            """
            )
            print("      ✅ statutes_fulltext_idx")

            session.run(
                """
                CREATE FULLTEXT INDEX precedents_fulltext_idx IF NOT EXISTS
                FOR (n:Precedent)
                ON EACH [n.title, n.summary]
            """
            )
            print("      ✅ precedents_fulltext_idx")

            # -----------------------------------------------------------------
            # STRUTTURA GRAFO (Codice -> Libro)
            # -----------------------------------------------------------------
            print("\n   🏛️ Creazione struttura grafo...")

            # Codici
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
            session.run(
                """
                MERGE (c:Codice {name: 'Codice Amministrativo'})
                SET c.description = 'Legge 7 agosto 1990, n. 241'
            """
            )

            # Libri Codice Penale (prefisso CP per distinguerli)
            libri_penale = [
                ("CP Libro I", "Dei reati in generale"),
                ("CP Libro II", "Dei delitti in particolare"),
                ("CP Libro III", "Delle contravvenzioni in particolare"),
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

            # Libri Codice Civile (prefisso CC per distinguerli)
            libri_civile = [
                ("CC Libro I", "Delle persone e della famiglia"),
                ("CC Libro II", "Delle successioni"),
                ("CC Libro III", "Della proprietà"),
                ("CC Libro IV", "Delle obbligazioni"),
                ("CC Libro V", "Del lavoro"),
                ("CC Libro VI", "Della tutela dei diritti"),
                ("CC Fuori range", "Articoli fuori dalla numerazione standard"),
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

            print("      ✅ Codice Penale (3 libri)")
            print("      ✅ Codice Civile (7 libri)")
            print("      ✅ Codice Amministrativo (senza libri)")

        print("\n✅ Schema creato")

    # =========================================================================
    # STEP 3: CARICAMENTO STATUTI
    # =========================================================================

    def load_statutes(self):
        """Carica Codice Penale, Civile e Amministrativo con embeddings."""
        from data_loader import (
            load_codice_amministrativo_with_embeddings,
            load_codice_civile_with_embeddings,
            load_codice_penale_with_embeddings,
        )

        print("\n📖 Caricamento statuti...")

        # Replace mode: avoid duplicates on repeated --statutes runs.
        with self.driver.session() as session:
            session.run("MATCH (s:Statute) DETACH DELETE s")
        print("   🧹 Nodi Statute esistenti rimossi")

        # Codice Penale
        print("\n   📕 Codice Penale...")
        df_penale, emb_penale = load_codice_penale_with_embeddings()
        self._ingest_statutes(df_penale, "codice_penale", emb_penale)

        # Codice Civile
        print("\n   📗 Codice Civile...")
        df_civile, emb_civile = load_codice_civile_with_embeddings()
        self._ingest_statutes(df_civile, "codice_civile", emb_civile)

        # Codice Amministrativo
        print("\n   📘 Codice Amministrativo (L. 241/1990)...")
        df_amm, emb_amm = load_codice_amministrativo_with_embeddings()
        self._ingest_statutes(df_amm, "codice_amministrativo", emb_amm)

        print("\n✅ Statuti caricati")

    def _ingest_statutes(self, df, source: str, embeddings):
        """Inserisce statuti nel database."""
        if embeddings is None:
            print(f"      ⚠️ Nessun embedding per {source}")
            return

        if len(embeddings) != len(df):
            print(
                f"      ⚠️ Mismatch: {len(df)} articoli vs {len(embeddings)} embeddings"
            )
            return

        with self.driver.session() as session:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                batch_emb = embeddings[i : i + BATCH_SIZE]

                records = []
                for idx, (_, row) in enumerate(batch.iterrows()):
                    global_idx = i + idx

                    # Normalizza campi
                    if source == "codice_civile":
                        articolo = str(row.get("article_id", ""))
                        titolo = str(row.get("article_title", ""))
                        testo = str(row.get("article_text", ""))
                    elif source == "codice_amministrativo":
                        articolo = str(row.get("numero", ""))
                        titolo = str(row.get("titolo", ""))
                        testo = str(row.get("contenuto", ""))
                    else:
                        articolo = str(row.get("articolo", ""))
                        titolo = str(row.get("titolo", ""))
                        testo = str(row.get("testo", ""))

                    def _clean_text(value) -> str:
                        if value is None:
                            return ""
                        as_str = str(value).strip()
                        return "" if as_str.lower() == "nan" else as_str

                    articolo = _clean_text(articolo)
                    titolo = _clean_text(titolo)
                    testo = _clean_text(testo)

                    libro = _clean_text(row.get("libro", ""))
                    if not libro:
                        libro = _clean_text(row.get("libro_codice_penale", ""))
                    if not libro:
                        libro = _clean_text(row.get("libro_codice_civile", ""))

                    records.append(
                        {
                            "statute_id": f"{source}_{global_idx}_{articolo}",
                            "articolo": articolo,
                            "titolo": titolo,
                            "testo": testo,
                            "libro": libro,
                            "source": source,
                            "full_text": f"Art. {articolo} - {titolo}: {testo}",
                            "embedding": batch_emb[idx].tolist(),
                        }
                    )

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
                    FOREACH (_ IN CASE WHEN record.libro <> '' THEN [1] ELSE [] END |
                        MERGE (l:Libro {name: record.libro, codice: record.source})
                        MERGE (s)-[:BELONGS_TO]->(l)
                    )
                """,
                    records=records,
                )

                inserted = min(i + BATCH_SIZE, len(df))
                print(f"      Inseriti: {inserted}/{len(df)}", end="\r")

            print(f"      ✅ {len(df)} articoli con embeddings")

    # =========================================================================
    # STEP 4: CARICAMENTO PRECEDENTI (itacasehold)
    # =========================================================================

    def load_precedents(self):
        """Carica precedenti itacasehold senza embeddings (fulltext only)."""
        from data_loader import load_itacasehold_metadata

        print("\n⚖️ Caricamento precedenti (itacasehold, no-embedding)...")

        try:
            metadata = load_itacasehold_metadata()
        except FileNotFoundError as e:
            print(f"      ⚠️ {e}")
            return

        with self.driver.session() as session:
            for i in range(0, len(metadata), BATCH_SIZE):
                batch_meta = metadata[i : i + BATCH_SIZE]

                records = []
                for idx, meta in enumerate(batch_meta):
                    global_idx = i + idx
                    records.append(
                        {
                            "precedent_id": f"prec_{global_idx}",
                            "title": str(meta.get("title", ""))[:500],
                            "summary": str(meta.get("summary", ""))[:5000],
                            "url": str(meta.get("url", "")),
                            "materia": str(meta.get("materia", ""))[:200],
                            "source": "itacasehold",
                        }
                    )

                session.run(
                    """
                    UNWIND $records AS record
                    CREATE (p:Precedent {
                        precedent_id: record.precedent_id,
                        title: record.title,
                        summary: record.summary,
                        url: record.url,
                        materia: record.materia,
                        source: record.source
                    })
                """,
                    records=records,
                )

                inserted = min(i + BATCH_SIZE, len(metadata))
                print(f"      Inseriti: {inserted}/{len(metadata)}", end="\r")

            print(f"      ✅ {len(metadata)} precedenti (senza embeddings)")

        print("\n✅ Precedenti caricati")

    # =========================================================================
    # STEP 5: VERIFICA INDICI
    # =========================================================================

    def wait_for_indexes(self, timeout: int = 120):
        """Attende che tutti gli indici siano ONLINE."""
        print("\n⏳ Attesa indici online...")

        start = time.time()
        while time.time() - start < timeout:
            with self.driver.session() as session:
                result = session.run(
                    """
                    SHOW INDEXES YIELD name, state
                    WHERE state <> 'ONLINE'
                    RETURN count(*) as pending
                """
                )
                pending = result.single()["pending"]

                if pending == 0:
                    print("   ✅ Tutti gli indici sono ONLINE")
                    return True

                print(f"   ⏳ {pending} indici in attesa...", end="\r")
                time.sleep(2)

        print("   ⚠️ Timeout - alcuni indici potrebbero non essere pronti")
        return False

    # =========================================================================
    # ORCHESTRAZIONE PRINCIPALE
    # =========================================================================

    def run_full_setup(
        self,
        clean: bool = False,
        load_statutes: bool = True,
        load_precedents: bool = True,
    ):
        """Esegue il setup completo del database.

        Args:
            clean: Se True, pulisce il database prima di inizializzare
            load_statutes: Se True, carica gli statuti (Codice Penale + Civile)
            load_precedents: Se True, carica i precedenti (itacasehold)
        """
        print("\n" + "=" * 60)
        print("🚀 LexCausa Database Orchestrator")
        print("=" * 60)

        start_time = time.time()

        # Step 0: Pulizia (opzionale)
        if clean:
            self.clean_database()

        # Step 1: Stato iniziale
        print("\n📊 Stato iniziale:")
        self.print_status()

        # Step 2: Schema (sempre necessario)
        self.create_schema()

        # Step 3: Statuti (opzionale)
        if load_statutes:
            self.load_statutes()
        else:
            print("\n📖 Caricamento statuti: SKIP")

        # Step 4: Precedenti (opzionale)
        if load_precedents:
            self.load_precedents()
        else:
            print("\n⚖️ Caricamento precedenti: SKIP")

        # Step 5: Attendi indici
        self.wait_for_indexes()

        # Stato finale
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("📊 Stato finale:")
        self.print_status()

        print(f"\n⏱️ Tempo totale: {elapsed:.1f}s")
        print("✅ Setup completato!")


def main():
    parser = argparse.ArgumentParser(
        description="LexCausa Database Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python db_orchestrator.py                    # Setup completo
  python db_orchestrator.py --clean            # Pulisce e reinizializza tutto
  python db_orchestrator.py --check            # Solo verifica stato DB
  python db_orchestrator.py --statutes         # Ricarica solo statuti
  python db_orchestrator.py --precedents       # Ricarica solo precedenti
  python db_orchestrator.py --clean --statutes # Pulisce e carica solo statuti
        """,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Pulisce il database prima di inizializzare",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Solo verifica stato database",
    )
    parser.add_argument(
        "--statutes",
        action="store_true",
        help="Carica solo statuti (Codice Penale + Civile)",
    )
    parser.add_argument(
        "--precedents",
        action="store_true",
        help="Carica solo precedenti (itacasehold)",
    )
    args = parser.parse_args()

    orchestrator = DatabaseOrchestrator()

    try:
        if args.check:
            orchestrator.print_status()
        else:
            # Se nessun flag specifico, carica tutto
            # Se almeno uno è specificato, carica solo quello
            if not args.statutes and not args.precedents:
                load_statutes = True
                load_precedents = True
            else:
                load_statutes = args.statutes
                load_precedents = args.precedents

            orchestrator.run_full_setup(
                clean=args.clean,
                load_statutes=load_statutes,
                load_precedents=load_precedents,
            )
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
