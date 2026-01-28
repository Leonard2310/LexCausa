#!/usr/bin/env python3
"""
LexCausa - Flask API Server

REST API server con logica corretta per la pipeline completa.
Il backend gestisce l'intero flusso: Reasoner → CounterReasoner.
"""

import os
import sys
import warnings
import json

from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# Setup paths
src_path = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(src_path) == "src":
    project_root = os.path.dirname(src_path)
else:
    project_root = src_path
    src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
os.chdir(project_root)

from agents import CounterReasoner, PolisherEvaluator, Reasoner  # noqa: E402
from agents.tools.neo4j_tools import get_legal_search_pipeline  # noqa: E402
from config import settings  # noqa: E402
from services.claim_classifier import ClaimClassifier  # noqa: E402

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Agenti globali (lazy initialization)
classifier = None
reasoner = None
counter_reasoner = None
polisher_evaluator = None

# Carica la tassonomia una volta all'avvio
TAXONOMY = None


def load_taxonomy():
    """Carica la tassonomia di causalità dal file JSON."""
    global TAXONOMY
    if TAXONOMY is None:
        candidate_paths = [
            settings.taxonomy_path,
            settings.data_dir / "tassonomia_causalita.json",
            settings.data_dir / "tassonomia_causale.json",
        ]

        taxonomy_path = next((p for p in candidate_paths if p.exists()), None)

        if taxonomy_path:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                TAXONOMY = json.load(f)
            print(f"✅ Tassonomia caricata da: {taxonomy_path}")
        else:
            print("⚠️ Tassonomia non trovata in nessun percorso noto")
            TAXONOMY = {"tassonomia_causalita": []}
    return TAXONOMY


def get_pipeline():
    """Get the shared LegalSearchPipeline singleton."""
    return get_legal_search_pipeline()


def get_classifier():
    """Lazy load del classificatore."""
    global classifier
    if classifier is None:
        print("🔧 Inizializzazione classificatore...")
        classifier = ClaimClassifier()
        print("✅ Classificatore pronto!")
    return classifier


def get_reasoner():
    """Lazy load del Reasoner agent."""
    global reasoner
    if reasoner is None:
        print("🔧 Inizializzazione Reasoner...")
        reasoner = Reasoner()
        print("✅ Reasoner pronto!")
    return reasoner


def get_counter_reasoner():
    """Lazy load del Counter-Reasoner agent."""
    global counter_reasoner
    if counter_reasoner is None:
        print("🔧 Inizializzazione Counter-Reasoner...")
        counter_reasoner = CounterReasoner()
        print("✅ Counter-Reasoner pronto!")
    return counter_reasoner


def get_polisher_evaluator():
    """Lazy load del Polisher-Evaluator agent."""
    global polisher_evaluator
    if polisher_evaluator is None:
        print("🔧 Inizializzazione Polisher-Evaluator...")
        polisher_evaluator = PolisherEvaluator()
        print("✅ Polisher-Evaluator pronto!")
    return polisher_evaluator


def get_warrant_causality(causality_type: str) -> dict:
    """
    Estrae il warrant dalla tassonomia per un dato tipo di causalità.
    
    Args:
        causality_type: Il tipo di causalità (es. "Materiale", "Giuridica", "Concause / Sopravvenute")
    
    Returns:
        Dict contenente:
        - warrant: dict con denominazione e todo_nli
        - attacking_causalities: lista delle causalità che possono attaccare questa
    """
    taxonomy = load_taxonomy()
    
    for entry in taxonomy.get("tassonomia_causalita", []):
        if entry.get("tipo_causalita") == causality_type:
            warrant = entry.get("warrant", {})
            
            # Il warrant contiene la "denominazione" che indica quale tipo di causalità può attaccare
            # Esempio: "Causalità Necessaria" può essere attaccata da "Causalità Sufficiente"
            attacking_causalities = []
            
            # Logica per determinare le causalità attaccanti basata sul warrant
            warrant_denominazione = warrant.get("denominazione", "")
            
            if "Necessaria" in warrant_denominazione:
                # La causalità necessaria può essere attaccata da cause sufficienti
                attacking_causalities.append("Concause / Sopravvenute")
            elif "Sufficiente Indipendente" in warrant_denominazione:
                # La causalità sufficiente indipendente può essere attaccata da cause necessarie
                attacking_causalities.append("Materiale")
            elif "Sufficiente (non da sola)" in warrant_denominazione:
                # Le concause possono essere attaccate da entrambe
                attacking_causalities.extend(["Materiale", "Giuridica"])
            
            return {
                "warrant": warrant,
                "attacking_causalities": attacking_causalities
            }
    
    return {"warrant": {}, "attacking_causalities": []}


def get_causality_details(causality_type: str) -> dict:
    """
    Recupera tutti i dettagli di una causalità dalla tassonomia.
    
    Args:
        causality_type: Il tipo di causalità
    
    Returns:
        Dict con descrizione, norme core, norme accessorie, ecc.
    """
    taxonomy = load_taxonomy()
    
    for entry in taxonomy.get("tassonomia_causalita", []):
        if entry.get("tipo_causalita") == causality_type:
            return {
                "tipo": entry.get("tipo_causalita"),
                "warrant": entry.get("warrant", {}),
                "descrizione": entry.get("descrizione_ruolo", ""),
                "principio": entry.get("principio_test_applicato", ""),
                "limiti": entry.get("limiti_criticita", ""),
                "norme_core": entry.get("norme_core", []),
                "norme_accessorie": entry.get("norme_accessorie", []),
                "note": entry.get("note", {})
            }
    
    return {}


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "LexCausa API", "version": "0.2.0"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint principale per il chatbot (Tab Ricerca).
    """
    try:
        pipe = get_pipeline()

        data = request.get_json()
        claim = data.get("message", "").strip()
        top_k = data.get("top_k", 5)

        if not claim:
            return jsonify({"error": 'Campo "message" obbligatorio'}), 400

        result = pipe.search(claim, top_k=top_k)
        response_text = format_search_result(result)

        return jsonify(
            {
                "response": response_text,
                "classification": {
                    "categories": result.classification.categories,
                    "descriptions": result.classification.descriptions,
                    "libro_mappings": result.classification.libro_mappings,
                    "sections": result.classification.sections,
                    "section_mappings": result.classification.section_mappings,
                },
                "articles": [
                    {
                        "source": art.source,
                        "articolo": art.articolo,
                        "titolo": art.titolo,
                        "testo": art.testo,
                        "libro": art.libro,
                        "score": art.score,
                    }
                    for art in result.articles
                ],
                "precedents": [],
            }
        )

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/reason", methods=["POST"])
def reason():
    """
    Endpoint per il ragionamento causale (Tab Ragionamento).
    """
    try:
        data = request.get_json()
        claim = data.get("claim", data.get("message", "")).strip()
        include_precedents = data.get("include_precedents", True)
        use_context = data.get("use_context", False)

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        reas = get_reasoner()

        if use_context:
            pipe = get_pipeline()
            search_result = pipe.search(claim, top_k=5)

            statutes = [
                {
                    "statute_id": art.statute_id,
                    "articolo": art.articolo,
                    "titolo": art.titolo,
                    "testo": art.testo,
                    "libro": art.libro,
                    "source": art.source,
                }
                for art in search_result.articles
            ]

            result = reas.reason_with_context(
                claim=claim,
                pre_retrieved_statutes=statutes,
                pre_retrieved_precedents=[],
            )
        else:
            result = reas.run(
                claim=claim,
                include_precedents=include_precedents,
            )

        return jsonify(
            {
                "claim": result.claim,
                "causality": result.causality_classification,
                "arguments": result.arguments,
                "reasoning_chain": result.reasoning_chain,
                "statutes": result.relevant_statutes,
                "precedents": result.relevant_precedents,
                "raw_response": result.raw_response,
            }
        )

    except Exception as e:
        print(f"❌ Errore reasoning: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/counter_reason", methods=["POST"])
def counter_reason():
    """
    Endpoint per il contro-ragionamento.
    
    Riceve:
    - claim: il claim legale
    - causality: la classificazione di causalità dal Reasoner
    
    Restituisce contro-argomenti basati sul warrant della causalità.
    """
    try:
        data = request.get_json()
        claim = data.get("claim", "").strip()
        causality = data.get("causality", {})
        include_precedents = data.get("include_precedents", True)
        max_statutes = data.get("max_statutes", 5)
        max_precedents = data.get("max_precedents", 3)

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400
        
        if not causality:
            return jsonify({"error": 'Campo "causality" obbligatorio'}), 400

        cr = get_counter_reasoner()
        
        # Esegui il counter-reasoning
        result = cr.run(
            claim=claim,
            causality=causality,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )

        return jsonify(result.to_dict())

    except Exception as e:
        print(f"❌ Errore counter-reasoning: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    """
    Endpoint per la pipeline completa (Tab Pipeline Completa).
    
    Gestisce l'intero flusso nel backend:
    1. Reasoner analizza il claim e produce causalità + argomenti
    2. CounterReasoner usa la causalità del Reasoner per generare contro-argomenti
    3. Restituisce entrambi i risultati al frontend
    """
    try:
        data = request.get_json()
        claim = data.get("claim", "").strip()
        include_precedents = data.get("include_precedents", True)
        max_statutes = data.get("max_statutes", 5)
        max_precedents = data.get("max_precedents", 3)

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        print(f"\n{'='*70}")
        print(f"🚀 PIPELINE COMPLETA - INIZIO")
        print(f"{'='*70}")
        print(f"Claim: {claim[:100]}...")
        
        # STEP 1: Reasoner
        print(f"\n{'─'*70}")
        print("📊 STEP 1: Esecuzione Reasoner...")
        print(f"{'─'*70}")
        
        reas = get_reasoner()
        reasoner_result = reas.run(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )
        
        print(f"✅ Reasoner completato")
        print(f"   - Causalità: {reasoner_result.causality_classification.get('causality_type', 'N/A')}")
        print(f"   - Argomenti: {len(reasoner_result.arguments)}")
        print(f"   - Statuti: {len(reasoner_result.relevant_statutes)}")
        print(f"   - Precedenti: {len(reasoner_result.relevant_precedents)}")

        # STEP 2: Counter-Reasoner
        print(f"\n{'─'*70}")
        print("⚔️  STEP 2: Esecuzione Counter-Reasoner...")
        print(f"{'─'*70}")
        
        cr = get_counter_reasoner()
        counter_result = cr.run(
            claim=claim,
            causality=reasoner_result.causality_classification,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )
        
        print(f"✅ Counter-Reasoner completato")
        print(f"   - Contro-argomenti: {len(counter_result.counter_arguments)}")
        print(f"   - Statuti: {len(counter_result.relevant_statutes)}")
        print(f"   - Precedenti: {len(counter_result.relevant_precedents)}")

        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETA - FINE")
        print(f"{'='*70}\n")

        # Restituisci entrambi i risultati
        return jsonify(
            {
                "claim": claim,
                "reasoner": reasoner_result.to_dict(),
                "counter_reasoner": counter_result.to_dict(),
            }
        )

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERRORE PIPELINE")
        print(f"{'='*70}")
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Endpoint per la valutazione finale.
    TODO: Implementare quando Polisher-Evaluator sarà completo.
    """
    try:
        data = request.get_json()
        claim = data.get("claim", "").strip()
        reasoner_output = data.get("reasoner_output", {})
        counter_output = data.get("counter_output", {})

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        pe = get_polisher_evaluator()
        result = pe.run(
            claim=claim,
            reasoner_output=reasoner_output,
            counter_reasoner_output=counter_output,
        )

        return jsonify(result.to_dict())

    except Exception as e:
        print(f"❌ Errore evaluation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def format_search_result(result) -> str:
    """Formatta il risultato in markdown."""
    lines = []

    lines.append("## 📋 Classificazione\n")
    for i, (cat, desc) in enumerate(
        zip(result.classification.categories, result.classification.descriptions), 1
    ):
        source, libro = result.classification.libro_mappings[i - 1]
        source_label = "Codice Civile" if "civile" in source else "Codice Penale"
        lines.append(f"**{i}. {cat}**")
        lines.append(f"- {desc}")
        lines.append(f"- 📚 [{source_label}] {libro}\n")

    if result.articles:
        lines.append("\n---\n")
        lines.append("## 📖 Articoli Rilevanti\n")
        for i, art in enumerate(result.articles, 1):
            src = "CC" if "civile" in art.source else "CP"
            lines.append(f"### {i}. [{src}] Art. {art.articolo} - {art.titolo}")
            lines.append(f"**Rilevanza:** {art.score:.1%}")
            lines.append(f"**Libro:** {art.libro}\n")
            preview = art.testo[:300] + "..." if len(art.testo) > 300 else art.testo
            lines.append(f"{preview}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  🚀 LexCausa API Server")
    print("=" * 70)
    print()
    print(f"  Server in ascolto su: http://{settings.api_host}:{settings.api_port}")
    print()
    print("  Endpoints:")
    print("  • GET  /health              - Health check")
    print("  • POST /api/chat            - Ricerca legale (Tab Ricerca)")
    print("  • POST /api/reason          - Ragionamento causale (Tab Ragionamento)")
    print("  • POST /api/counter_reason  - Contro-ragionamento")
    print("  • POST /api/pipeline        - Pipeline completa (NUOVO)")
    print("  • POST /api/evaluate        - Valutazione finale (stub)")
    print()
    print("=" * 70)
    print()

    app.run(host=settings.api_host, port=settings.api_port, debug=settings.debug)
