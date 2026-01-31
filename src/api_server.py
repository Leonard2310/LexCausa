#!/usr/bin/env python3
"""
LexCausa - Flask API Server

REST API server con logica corretta per la pipeline completa.
Il backend gestisce l'intero flusso: Reasoner  CounterReasoner.
"""

import json
import os
import sys
import warnings

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
from agents.router import Router, RoutingDecision  # noqa: E402
from agents.tools import config_loader  # noqa: E402
from agents.tools.neo4j_tools import (  # noqa: E402
    get_legal_search_pipeline,
    search_precedents_tool,
)
from config import settings  # noqa: E402
from services.claim_classifier import ClaimClassifier  # noqa: E402
from services.stance_classifier import StanceClassifier  # noqa: E402

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Agenti globali (lazy initialization)
classifier = None
reasoner = None
counter_reasoner = None
polisher_evaluator = None
stance_classifier = None
router_agent = None

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
        else:
            print("⚠️ Tassonomia non trovata in nessun percorso noto")
            TAXONOMY = {"tassonomia_causalita": []}
    return TAXONOMY


def get_pipeline():
    """Get the shared LegalSearchPipeline singleton."""
    return get_legal_search_pipeline()


def prepare_claim_context(
    claim: str,
    include_precedents: bool,
    max_statutes: int,
    max_precedents: int,
) -> tuple[list[dict], list[dict]]:
    """Pre-retrieve statutes and precedents before reasoning."""
    print(
        f"🔎 Pre-retrieval config: top_k_statutes={max_statutes}, max_precedents={max_precedents}"
    )
    pipe = get_pipeline()
    search_result = pipe.search(claim, top_k=max_statutes)

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

    reas = get_reasoner()
    statutes = reas.filter_irrelevant_statutes(claim, statutes)

    precedents: list[dict] = []
    if include_precedents:
        try:
            result = search_precedents_tool.invoke(
                {"query": claim, "limit": max_precedents}
            )
            if isinstance(result, list):
                precedents = result
        except Exception as e:
            print(f"⚠️ Errore recupero precedenti: {e}")

    precedents = reas.filter_irrelevant_precedents(claim, precedents)

    return statutes, precedents


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


def get_router():
    """Lazy load del Router preliminare."""
    global router_agent
    if router_agent is None:
        print("🔧 Inizializzazione Router...")
        router_agent = Router()
        print("✅ Router pronto!")
    return router_agent


def get_polisher_evaluator():
    """Lazy load del Polisher-Evaluator agent."""
    global polisher_evaluator
    if polisher_evaluator is None:
        print("🔧 Inizializzazione Polisher-Evaluator...")
        polisher_evaluator = PolisherEvaluator()
        print("✅ Polisher-Evaluator pronto!")
    return polisher_evaluator


def get_stance_classifier():
    """Lazy load dello Stance Classifier NLI."""
    global stance_classifier
    if stance_classifier is None:
        print("🔧 Inizializzazione Stance Classifier (NLI)...")
        stance_classifier = StanceClassifier()
        print("✅ Stance Classifier pronto!")
    return stance_classifier


def resolve_routing_decision(
    claim: str, payload: dict | None = None
) -> RoutingDecision:
    """
    Determina il routing (causal_type_id/theory_id) usando eventuali hint del payload,
    altrimenti invoca il Router.
    """
    router = get_router()
    payload = payload or {}
    routing_hint = payload.get("routing") or payload.get("causality") or payload

    ct = (
        routing_hint.get("causal_type_id")
        or routing_hint.get("causality_type")
        or routing_hint.get("causal_type")
    )
    th = routing_hint.get("theory_id")

    if ct:
        ct_valid, th_valid = config_loader.validate_ids(ct, th)
        return RoutingDecision(
            claim=claim,
            causal_type_id=ct_valid,
            theory_id=th_valid or "",
            anchor_norms=config_loader.anchor_norms_for(ct_valid),
            principle_tests=config_loader.principle_tests_for(ct_valid),
        )

    return router.route(claim)


def classify_stance_for_agents(
    claim: str,
    statutes: list[dict],
    precedents: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Classify statutes and precedents as supporting or opposing the claim.

    Returns:
        Tuple of (support_statutes, against_statutes, support_precedents, against_precedents)
    """
    sc = get_stance_classifier()

    print(f"\n{'─'*70}")
    print("🎯 STANCE CLASSIFICATION (NLI)...")
    print(f"{'─'*70}")

    support_statutes, against_statutes, neutral_statutes = sc.classify_statutes_batch(
        claim, statutes
    )
    support_precedents, against_precedents, neutral_precedents = (
        sc.classify_precedents_batch(claim, precedents)
    )

    # Re-introduce neutrals to both agents to avoid starving them of context
    support_statutes = support_statutes + neutral_statutes
    against_statutes = against_statutes + neutral_statutes
    support_precedents = support_precedents + neutral_precedents
    against_precedents = against_precedents + neutral_precedents

    print("\n📊 Risultato stance classification:")
    print("   - Articoli a SUPPORTO: " + str(len(support_statutes)))
    print("   - Articoli CONTRO: " + str(len(against_statutes)))
    print("   - Precedenti a SUPPORTO: " + str(len(support_precedents)))
    print("   - Precedenti CONTRO: " + str(len(against_precedents)))

    return support_statutes, against_statutes, support_precedents, against_precedents


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

            return {"warrant": warrant, "attacking_causalities": attacking_causalities}

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
                "note": entry.get("note", {}),
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
        top_k = data.get("top_k", settings.search_top_k_default)

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
        max_statutes = data.get("max_statutes", settings.search_top_k_default)
        max_precedents = data.get("max_precedents", settings.precedents_limit_default)
        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        routing_decision = resolve_routing_decision(claim, data)
        reas = get_reasoner()
        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )

        result = reas.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=statutes,
            pre_retrieved_precedents=precedents,
        )

        return jsonify(
            {
                "claim": result.claim,
                "causality": result.causality_classification,
                "routing": routing_decision.to_dict(),
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
    - (opzionale) causal_type_id/theory_id: se assenti, vengono scelti dal Router

    Restituisce contro-argomenti basati sulla config di causalità.
    """
    try:
        data = request.get_json()
        claim = data.get("claim", "").strip()
        include_precedents = data.get("include_precedents", True)
        max_statutes = data.get("max_statutes", settings.search_top_k_default)
        max_precedents = data.get("max_precedents", settings.precedents_limit_default)

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        routing_decision = resolve_routing_decision(claim, data)
        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )

        # Classifica stance per fornire al counter norme contrarie/neutral
        _, against_statutes, _, against_precedents = classify_stance_for_agents(
            claim, statutes, precedents
        )

        cr = get_counter_reasoner()

        # Esegui il counter-reasoning con contesto pre-retrieved
        result = cr.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=against_statutes,
            pre_retrieved_precedents=against_precedents,
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
        max_statutes = data.get("max_statutes", settings.search_top_k_default)
        max_precedents = data.get("max_precedents", settings.precedents_limit_default)

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        print(f"\n{'='*70}")
        print("🚀 PIPELINE COMPLETA - INIZIO")
        print(f"{'='*70}")
        print(f"Claim: {claim[:100]}...")

        routing_decision = resolve_routing_decision(claim, data)

        # Preload context once for both Reasoner and Counter-Reasoner
        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )

        # Classify stance using NLI to separate support vs against
        support_statutes, against_statutes, support_precedents, against_precedents = (
            classify_stance_for_agents(claim, statutes, precedents)
        )

        # STEP 1: Reasoner (receives SUPPORT articles/precedents)
        print(f"\n{'─'*70}")
        print("📊 STEP 1: Esecuzione Reasoner (con articoli a SUPPORTO)...")
        print(f"{'─'*70}")
        print(
            f"   📚 Knowledge base: {len(support_statutes)} statuti, {len(support_precedents)} precedenti"
        )

        reas = get_reasoner()
        reasoner_result = reas.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=support_statutes,
            pre_retrieved_precedents=support_precedents,
        )

        print("✅ Reasoner completato")
        print(
            f"   - Causalità: {routing_decision.causal_type_id} / {routing_decision.theory_id}"
        )
        print(f"   - Argomenti: {len(reasoner_result.arguments)}")
        print(
            f"   - Catena di ragionamento: {len(reasoner_result.reasoning_chain)} steps"
        )

        # STEP 2: Counter-Reasoner (receives AGAINST articles/precedents)
        print(f"\n{'─'*70}")
        print("⚔️  STEP 2: Esecuzione Counter-Reasoner (con articoli CONTRO)...")
        print(f"{'─'*70}")
        print(
            f"   📚 Knowledge base: {len(against_statutes)} statuti, {len(against_precedents)} precedenti"
        )

        cr = get_counter_reasoner()
        counter_result = cr.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=against_statutes,
            pre_retrieved_precedents=against_precedents,
        )

        print("✅ Counter-Reasoner completato")
        print(f"   - Contro-argomenti: {len(counter_result.counter_arguments)}")
        print(
            f"   - Catena di contro-ragionamento: {len(counter_result.reasoning_chain)} steps"
        )

        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETA - FINE")
        print(f"{'='*70}\n")

        # Restituisci entrambi i risultati
        return jsonify(
            {
                "claim": claim,
                "routing": routing_decision.to_dict(),
                "reasoner": reasoner_result.to_dict(),
                "counter_reasoner": counter_result.to_dict(),
            }
        )

    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ ERRORE PIPELINE")
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
    print("  • POST /api/pipeline        - Pipeline completa")
    print("  • POST /api/evaluate        - Valutazione finale (stub)")
    print()
    print("=" * 70)
    print()

    app.run(host=settings.api_host, port=settings.api_port, debug=settings.debug)
