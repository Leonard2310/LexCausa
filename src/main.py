#!/usr/bin/env python3
"""
LexCausa - Main Orchestrator

Interactive CLI to test and run the complete legal search pipeline.
Allows testing individual components or the full pipeline.

Components:
1. ClaimClassifier - Classifies legal claims using Groq LLM
2. LegalSearchPipeline - Complete search with embeddings + Neo4j
3. Neo4j Health Check - Verify database status and indexes
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add src to path for imports
src_path = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(src_path) == "src":
    # Running from src/main.py
    project_root = os.path.dirname(src_path)
else:
    # Running from project root
    project_root = src_path
    src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
os.chdir(project_root)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def print_header():
    """Print the application header."""
    print()
    print("=" * 70)
    print(r"  _                ____                       ")
    print(r" | |    _____  __ / ___|__ _ _   _ ___  __ _  ")
    print(r" | |   / _ \ \/ /| |   / _` | | | / __|/ _` | ")
    print(r" | |__|  __/>  < | |__| (_| | |_| \__ \ (_| | ")
    print(r" |_____\___/_/\_\ \____\__,_|\__,_|___/\__,_| ")
    print()
    print("  Legal Search Pipeline - v0.1.0")
    print("=" * 70)
    print()


def print_menu():
    """Print the main menu."""
    print("\n" + "-" * 50)
    print("📋 MENU PRINCIPALE")
    print("-" * 50)
    print("  1. 🔍 Ricerca completa (Pipeline)")
    print("  2. 🏷️  Solo classificazione claim")
    print("  3. 📊 Test embedding generation")
    print("  4. 🗄️  Neo4j health check")
    print("  5. 📈 Statistiche database")
    print("  6. 🧪 Esegui test automatici")
    print("  7. ❓ Help")
    print("  0. 🚪 Esci")
    print("-" * 50)


def neo4j_health_check():
    """Check Neo4j connection and indexes."""
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")

    print("\n🔍 Neo4j Health Check")
    print("-" * 40)

    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))

        with driver.session() as session:
            # Connection test
            result = session.run("RETURN 1 as test")
            result.single()
            print(f"✅ Connessione: {uri}")

            # Check indexes
            result = session.run(
                "SHOW INDEXES YIELD name, type, state, labelsOrTypes, options"
            )
            print("\n📊 Indici vettoriali:")
            for rec in result:
                if "VECTOR" in rec["type"]:
                    dims = (
                        rec["options"]
                        .get("indexConfig", {})
                        .get("vector.dimensions", "?")
                    )
                    print(
                        f"   • {rec['name']}: {rec['state']} "
                        f"({dims} dim) -> {rec['labelsOrTypes']}"
                    )

            # Node counts
            print("\n📈 Conteggio nodi:")
            for label in ["Statute", "Precedent", "Libro"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as c")
                count = result.single()["c"]
                print(f"   • {label}: {count}")

        driver.close()
        print("\n✅ Health check completato!")

    except Exception as e:
        print(f"❌ Errore connessione: {e}")


def database_statistics():
    """Show detailed database statistics."""
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")

    print("\n📈 Statistiche Database")
    print("-" * 40)

    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))

        with driver.session() as session:
            # Statutes by source
            print("\n📚 Articoli per codice:")
            result = session.run(
                """
                MATCH (s:Statute)
                RETURN s.source as source, count(*) as count
                ORDER BY count DESC
            """
            )
            for rec in result:
                source = rec["source"] or "N/A"
                print(f"   • {source}: {rec['count']} articoli")

            # Statutes by libro
            print("\n📖 Articoli per libro:")
            result = session.run(
                """
                MATCH (s:Statute)
                RETURN s.source as source, s.libro as libro, count(*) as count
                ORDER BY source, libro
            """
            )
            for rec in result:
                source = "CC" if "civile" in str(rec["source"]) else "CP"
                print(f"   • [{source}] {rec['libro']}: {rec['count']}")

            # Precedents by materia (top 10)
            print("\n⚖️  Top 10 precedenti per materia:")
            result = session.run(
                """
                MATCH (p:Precedent)
                RETURN p.materia as materia, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """
            )
            for rec in result:
                materia = rec["materia"] or "N/A"
                print(f"   • {materia}: {rec['count']} chunks")

            # Embedding coverage
            print("\n🧮 Copertura embeddings:")
            result = session.run(
                """
                MATCH (s:Statute)
                WHERE s.embedding IS NOT NULL
                RETURN count(*) as with_emb
            """
            )
            with_emb = result.single()["with_emb"]
            result = session.run("MATCH (s:Statute) RETURN count(*) as total")
            total = result.single()["total"]
            pct = (with_emb / total * 100) if total > 0 else 0
            print(f"   • Statute con embedding: {with_emb}/{total} ({pct:.1f}%)")

            result = session.run(
                """
                MATCH (p:Precedent)
                WHERE p.embedding IS NOT NULL
                RETURN count(*) as with_emb
            """
            )
            with_emb = result.single()["with_emb"]
            result = session.run("MATCH (p:Precedent) RETURN count(*) as total")
            total = result.single()["total"]
            pct = (with_emb / total * 100) if total > 0 else 0
            print(f"   • Precedent con embedding: {with_emb}/{total} ({pct:.1f}%)")

        driver.close()
        print("\n✅ Statistiche complete!")

    except Exception as e:
        print(f"❌ Errore: {e}")


def test_classification():
    """Test only the claim classification component."""
    from services.claim_classifier import ClaimClassifier

    print("\n🏷️  Test Classificatore")
    print("-" * 40)

    try:
        classifier = ClaimClassifier()
        print("✅ Classificatore inizializzato\n")

        claim = input("📝 Inserisci il claim da classificare:\n> ").strip()
        if not claim:
            print("⚠️  Claim vuoto, annullato.")
            return

        print("\n🔄 Classificazione in corso...")
        result = classifier.classify(claim)

        print("\n📊 Risultato classificazione:")
        print("-" * 40)
        for i, (cat, desc) in enumerate(zip(result.categories, result.descriptions), 1):
            source, libro = result.libro_mappings[i - 1]
            source_label = "Codice Civile" if "civile" in source else "Codice Penale"
            print(f"   {i}. {cat}")
            print(f"      → {desc}")
            print(f"      → [{source_label}] {libro}")
            print()

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback

        traceback.print_exc()


def test_embedding():
    """Test embedding generation."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print("\n📊 Test Embedding Generation")
    print("-" * 40)

    model_name = "nlpaueb/legal-bert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"🔧 Caricamento modello su {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        model = model.to(device)
        model.eval()
        print("✅ Modello caricato!\n")

        text = input("📝 Inserisci il testo da embeddare:\n> ").strip()
        if not text:
            print("⚠️  Testo vuoto, annullato.")
            return

        print("\n🔄 Generazione embedding...")
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        token_emb = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        embedding = (token_emb * mask).sum(1) / mask.sum(1)
        embedding = embedding.cpu().numpy()[0]

        print("\n✅ Embedding generato!")
        print(f"   • Dimensioni: {len(embedding)}")
        print(f"   • Min: {embedding.min():.4f}")
        print(f"   • Max: {embedding.max():.4f}")
        print(f"   • Mean: {embedding.mean():.4f}")
        print(f"   • Primi 5 valori: {embedding[:5].round(4).tolist()}")

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback

        traceback.print_exc()


def full_search():
    """Run the complete search pipeline."""
    from services.legal_search import LegalSearchPipeline

    print("\n🔍 Ricerca Completa")
    print("-" * 40)

    try:
        print("🔧 Inizializzazione pipeline...")
        pipeline = LegalSearchPipeline()
        print()

        while True:
            claim = input("📝 Inserisci il claim legale (o 'back' per tornare):\n> ")
            claim = claim.strip()

            if claim.lower() in ("back", "b", ""):
                break

            top_k = input("📊 Quanti risultati? [default: 5]: ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5

            print("\n" + "=" * 60)
            result = pipeline.search(claim, top_k=top_k)
            print()
            print(result)
            print()

        pipeline.close()

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback

        traceback.print_exc()


def run_automatic_tests():
    """Run a suite of automatic tests."""
    from services.legal_search import LegalSearchPipeline

    print("\n🧪 Test Automatici")
    print("-" * 40)

    test_claims = [
        {
            "claim": (
                "Il venditore non ha consegnato l'immobile nei tempi "
                "previsti dal contratto."
            ),
            "expected_type": "civile",
            "description": "Inadempimento contrattuale",
        },
        {
            "claim": (
                "L'imputato ha sottratto denaro dalla cassa del "
                "supermercato in cui lavorava."
            ),
            "expected_type": "penale",
            "description": "Furto/appropriazione indebita",
        },
        {
            "claim": (
                "Il testamento olografo del de cuius è stato impugnato "
                "dai legittimari per lesione di quota."
            ),
            "expected_type": "civile",
            "description": "Successioni",
        },
        {
            "claim": (
                "Il conducente ha provocato un incidente stradale "
                "causando lesioni gravi al pedone."
            ),
            "expected_type": "penale",
            "description": "Lesioni colpose",
        },
        {
            "claim": (
                "Il locatore pretende il pagamento di canoni arretrati "
                "per 6 mensilità."
            ),
            "expected_type": "civile",
            "description": "Locazione",
        },
    ]

    try:
        print("🔧 Inizializzazione pipeline...")
        pipeline = LegalSearchPipeline()
        print()

        passed = 0
        failed = 0

        for i, test in enumerate(test_claims, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {test['description']}")
            print(f"{'='*60}")
            print(f"📝 Claim: {test['claim']}")
            print(f"📋 Tipo atteso: {test['expected_type']}")

            try:
                result = pipeline.search(test["claim"], top_k=10)

                # Check classification
                categories = result.classification.categories
                expected_prefix = "CC" if test["expected_type"] == "civile" else "CP"
                classification_ok = any(
                    cat.startswith(expected_prefix) for cat in categories
                )

                print(f"\n📊 Classificazione: {categories}")
                print(f"📚 Articoli trovati: {len(result.articles)}")

                if result.articles:
                    print("\n🏆 Top 3 risultati:")
                    for j, art in enumerate(result.articles[:3], 1):
                        src = "CC" if "civile" in art.source else "CP"
                        print(
                            f"   {j}. [{src}] Art. {art.articolo} - "
                            f"{art.titolo[:40]}... (score: {art.score:.4f})"
                        )

                if classification_ok:
                    print("\n✅ TEST PASSED")
                    passed += 1
                else:
                    print(f"\n❌ TEST FAILED - Expected {expected_prefix} in top 3")
                    failed += 1

            except Exception as e:
                print(f"\n❌ TEST ERROR: {e}")
                failed += 1

        print("\n" + "=" * 60)
        print("📊 RIEPILOGO TEST")
        print("=" * 60)
        print(f"   ✅ Passati: {passed}/{len(test_claims)}")
        print(f"   ❌ Falliti: {failed}/{len(test_claims)}")
        pct = (passed / len(test_claims) * 100) if test_claims else 0
        print(f"   📈 Success rate: {pct:.1f}%")
        print()

        pipeline.close()

    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        import traceback

        traceback.print_exc()


def show_help():
    """Show help information."""
    print("\n❓ HELP - LexCausa Pipeline")
    print("=" * 60)
    print(
        """
LexCausa è una pipeline di ricerca legale che permette di:

1. RICERCA COMPLETA
   Inserisci un claim legale e ottieni gli articoli più rilevanti
   dal Codice Civile e Penale italiano.

   La pipeline:
   a) Classifica il claim nei libri appropriati (via Groq LLM)
   b) Genera un embedding del claim (via Legal-BERT)
   c) Esegue ricerca vettoriale in Neo4j
   d) Restituisce gli articoli più simili

2. SOLO CLASSIFICAZIONE
   Testa solo il componente di classificazione LLM.
   Utile per verificare che il routing nei libri sia corretto.

3. TEST EMBEDDING
   Testa la generazione di embeddings con Legal-BERT.
   Mostra dimensioni e statistiche del vettore.

4. NEO4J HEALTH CHECK
   Verifica la connessione a Neo4j e lo stato degli indici.

5. STATISTICHE DATABASE
   Mostra conteggi dettagliati di nodi, articoli per libro,
   precedenti per materia, ecc.

6. TEST AUTOMATICI
   Esegue una suite di test con claim predefiniti per
   verificare il funzionamento end-to-end della pipeline.

VARIABILI D'AMBIENTE RICHIESTE:
   - NEO4J_URI: URI di connessione Neo4j (es: bolt://localhost:7687)
   - NEO4J_USER: Username Neo4j
   - NEO4J_PASSWORD: Password Neo4j
   - GROQ_API_KEY: API key per Groq Cloud
"""
    )


def main():
    """Main orchestrator entry point."""
    print_header()

    # Check environment
    required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GROQ_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print("⚠️  Variabili d'ambiente mancanti:")
        for v in missing:
            print(f"   • {v}")
        print("\nAssicurati di avere un file .env configurato.")
        print()

    while True:
        print_menu()

        try:
            choice = input("\n👉 Scegli un'opzione: ").strip()

            if choice == "0":
                print("\n👋 Arrivederci!")
                break
            elif choice == "1":
                full_search()
            elif choice == "2":
                test_classification()
            elif choice == "3":
                test_embedding()
            elif choice == "4":
                neo4j_health_check()
            elif choice == "5":
                database_statistics()
            elif choice == "6":
                run_automatic_tests()
            elif choice == "7":
                show_help()
            else:
                print("⚠️  Opzione non valida. Riprova.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrotto dall'utente. Arrivederci!")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")


if __name__ == "__main__":
    main()
