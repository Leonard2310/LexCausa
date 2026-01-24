#!/usr/bin/env python3
"""
LexCausa - Flask API Server (per frontend)

REST API server da aggiungere al progetto esistente.
Espone gli endpoint necessari per il frontend React.

Salva questo file come: src/api_server.py
"""

import os
import sys
import warnings

from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# Setup paths (mantieni la stessa logica del main.py)
src_path = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(src_path) == "src":
    project_root = os.path.dirname(src_path)
else:
    project_root = src_path
    src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
os.chdir(project_root)

from agents import CounterReasoner, PolisherEvaluator, Reasoner  # noqa: E402
from config import settings  # noqa: E402
from services.claim_classifier import ClaimClassifier  # noqa: E402
from services.legal_search import LegalSearchPipeline  # noqa: E402

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Pipeline e agenti globali (lazy initialization)
pipeline = None
classifier = None
reasoner = None
counter_reasoner = None
polisher_evaluator = None


def get_pipeline():
    """Lazy load della pipeline."""
    global pipeline
    if pipeline is None:
        print("🔧 Inizializzazione pipeline...")
        pipeline = LegalSearchPipeline()
        print("✅ Pipeline pronta!")
    return pipeline


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


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "LexCausa API", "version": "0.1.0"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint principale per il chatbot.

    Request: {"message": "claim legale", "top_k": 5}
    Response: {"response": "testo formattato", ...}
    """
    try:
        pipe = get_pipeline()

        data = request.get_json()
        claim = data.get("message", "").strip()
        top_k = data.get("top_k", 5)

        if not claim:
            return jsonify({"error": 'Campo "message" obbligatorio'}), 400

        # Esegui ricerca
        result = pipe.search(claim, top_k=top_k)

        # Formatta risposta
        response_text = format_search_result(result)

        return jsonify(
            {
                "response": response_text,
                "classification": {
                    "categories": result.classification.categories,
                    "descriptions": result.classification.descriptions,
                    "libro_mappings": result.classification.libro_mappings,
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
                "precedents": [],  # TODO: aggiungere ricerca precedenti
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
    Endpoint per il ragionamento causale.

    Request: {"claim": "claim legale", "include_precedents": true, "use_context": false}
    Response: {"causality": {...}, "arguments": [...], "reasoning_chain": [...]}
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
            # Prima recupera contesto dalla pipeline, poi ragiona
            pipe = get_pipeline()
            search_result = pipe.search(claim, top_k=5)

            # Converti articoli in formato dict
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
            # Ragionamento con tool calling
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


@app.route("/api/counter-reason", methods=["POST"])
def counter_reason():
    """
    Endpoint per il contro-ragionamento.

    Request: {"claim": "...", "causality_type": "...", "arguments": [...]}
    Response: {"counter_arguments": [...], "attack_graph": {...}}

    TODO: Implementare quando Counter-Reasoner sarà completo.
    """
    try:
        data = request.get_json()
        claim = data.get("claim", "").strip()
        causality_type = data.get("causality_type", "")
        arguments = data.get("arguments", [])

        if not claim or not causality_type:
            return (
                jsonify({"error": 'Campi "claim" e "causality_type" obbligatori'}),
                400,
            )

        cr = get_counter_reasoner()
        result = cr.run(
            claim=claim,
            causality_type=causality_type,
            reasoner_arguments=arguments,
        )

        return jsonify(result.to_dict())

    except Exception as e:
        print(f"❌ Errore counter-reasoning: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Endpoint per la valutazione finale.

    Request: {"claim": "...", "reasoner_output": {...}, "counter_output": {...}}
    Response: {"winning_side": "...", "summary": "...", "polished_response": "..."}

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

    # Classificazione
    lines.append("## 📋 Classificazione\n")
    for i, (cat, desc) in enumerate(
        zip(result.classification.categories, result.classification.descriptions), 1
    ):
        source, libro = result.classification.libro_mappings[i - 1]
        source_label = "Codice Civile" if "civile" in source else "Codice Penale"
        lines.append(f"**{i}. {cat}**")
        lines.append(f"- {desc}")
        lines.append(f"- 📚 [{source_label}] {libro}\n")

    # Articoli
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

    # Precedenti (TODO: da implementare nella pipeline)
    # if hasattr(result, 'precedents') and result.precedents:
    #     ...

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
    print("  • GET  /health            - Health check")
    print("  • POST /api/chat          - Ricerca legale (statuti)")
    print("  • POST /api/reason        - Ragionamento causale")
    print("  • POST /api/counter-reason - Contro-ragionamento (stub)")
    print("  • POST /api/evaluate      - Valutazione finale (stub)")
    print()
    print("=" * 70)
    print()

    app.run(host=settings.api_host, port=settings.api_port, debug=settings.debug)
