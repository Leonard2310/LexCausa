#!/usr/bin/env python3
"""
LexCausa - Flask API Server

REST API server con logica corretta per la pipeline completa.
Il backend gestisce l'intero flusso: Reasoner  CounterReasoner.
"""

import json
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from queue import Empty, Queue

from flask import Flask, Response, jsonify, request, stream_with_context
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
from agents.base import AgentConfig  # noqa: E402
from agents.router import Router, RoutingDecision  # noqa: E402
from agents.tools import config_loader  # noqa: E402
from agents.tools.neo4j_tools import (  # noqa: E402
    get_legal_search_pipeline,
    search_precedents_tool,
)
from config import settings  # noqa: E402
from services.stance_classifier import StanceClassifier  # noqa: E402

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Prevent concurrent pipeline executions (interleaves logs & shared state)
_pipeline_lock = threading.Lock()

# ─── Pipeline file logging ──────────────────────────────────────────
LOG_DIR = Path(project_root) / "logs"


class _TeeWriter:
    """Write to both the original stream and a StringIO buffer."""

    def __init__(self, original, buffer: StringIO):
        self._original = original
        self._buffer = buffer

    def write(self, text: str) -> int:
        self._original.write(text)
        self._buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self._original.flush()

    # delegate any other attribute to the original stream
    def __getattr__(self, name: str):
        return getattr(self._original, name)


@contextmanager
def _pipeline_logger(claim: str):
    """
    Context manager: captures all stdout produced during a pipeline run
    and saves it to ``logs/<timestamp>_<slug>.log``.
    """
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = claim[:60].replace(" ", "_").replace("/", "-")
    log_path = LOG_DIR / f"{ts}_{slug}.log"

    buf = StringIO()
    buf.write(f"[{datetime.now().isoformat()}] Pipeline log for claim:\n")
    buf.write(f"{claim}\n")
    buf.write("=" * 70 + "\n\n")

    old_stdout = sys.stdout
    sys.stdout = _TeeWriter(old_stdout, buf)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_stdout
        try:
            log_path.write_text(buf.getvalue(), encoding="utf-8")
            print(f"\n📝 Pipeline log salvato in: {log_path}")
        except Exception as exc:
            print(f"⚠️ Errore salvataggio log: {exc}")


def _slugify_filename(text: str, max_len: int = 60) -> str:
    clean = (text or "").strip().replace("\n", " ")
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in clean)
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe[:max_len] or "claim").strip("_")


def _persist_repaired_aspic_files(claim: str, evaluation_payload: dict) -> dict:
    """Persist repaired ASPIC+ IR JSON files (reasoner/counter) under logs/aspic_repairs."""
    if not isinstance(evaluation_payload, dict):
        return {}

    repaired_reasoner = evaluation_payload.get("repaired_reasoner_aspic_ir") or {}
    repaired_counter = evaluation_payload.get("repaired_counter_aspic_ir") or {}

    if not isinstance(repaired_reasoner, dict):
        repaired_reasoner = {}
    if not isinstance(repaired_counter, dict):
        repaired_counter = {}

    if not repaired_reasoner and not repaired_counter:
        return {}

    out_dir = LOG_DIR / "aspic_repairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify_filename(claim)
    files: dict[str, dict[str, str]] = {}

    def _write_json(role: str, payload: dict) -> None:
        filename = f"{ts}_{slug}_{role}_repaired_aspic_ir.json"
        path = out_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files[role] = {
            "absolute_path": str(path),
            "relative_path": str(path.relative_to(project_root)),
        }

    try:
        if repaired_reasoner:
            _write_json("reasoner", repaired_reasoner)
        if repaired_counter:
            _write_json("counter_reasoner", repaired_counter)
    except Exception as exc:
        print(f"⚠️ Errore salvataggio ASPIC+ riparato: {exc}")
        return {}

    if files:
        print(f"🧩 ASPIC+ riparati salvati: {files}")
    return files


def _persist_aqa_report_file(claim: str, evaluation_payload: dict) -> dict:
    """Persist AQA report JSON under logs/aqa_reports."""
    if not isinstance(evaluation_payload, dict):
        return {}

    aqa_report = evaluation_payload.get("aqa_report")
    if not isinstance(aqa_report, dict) or not aqa_report:
        return {}

    out_dir = LOG_DIR / "aqa_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify_filename(claim)
    filename = f"{ts}_{slug}_aqa_report.json"
    path = out_dir / filename

    try:
        path.write_text(
            json.dumps(aqa_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"⚠️ Errore salvataggio report AQA: {exc}")
        return {}

    payload = {
        "absolute_path": str(path),
        "relative_path": str(path.relative_to(project_root)),
    }
    print(f"📊 Report AQA salvato: {payload}")
    return payload


# Agenti globali (lazy initialization)
reasoner = None
counter_reasoner = None
polisher_evaluator = None
stance_classifier = None
router_agent = None


def get_pipeline():
    """Get the shared LegalSearchPipeline singleton."""
    return get_legal_search_pipeline()


def _articles_to_dicts(articles) -> list[dict]:
    """Convert ArticleResult objects to plain dicts."""
    return [
        {
            "statute_id": art.statute_id,
            "articolo": art.articolo,
            "titolo": art.titolo,
            "testo": art.testo,
            "libro": art.libro,
            "source": art.source,
        }
        for art in articles
    ]


def _article_with_retrieval_debug(art) -> dict:
    """Serialize an ArticleResult including hybrid retrieval score breakdown."""
    breakdown = art.score_debug or {}
    return {
        "statute_id": art.statute_id,
        "source": art.source,
        "libro": art.libro,
        "articolo": art.articolo,
        "titolo": art.titolo,
        "testo": art.testo,
        "score": float(art.score),
        "vector_rank_score": float(breakdown.get("vector_rank_score", 0.0)),
        "fulltext_rank_score": float(breakdown.get("fulltext_rank_score", 0.0)),
        "fusion_score": float(breakdown.get("fusion_score", 0.0)),
        "keyword_bonus": float(breakdown.get("keyword_bonus", 0.0)),
        "priority_multiplier": float(breakdown.get("priority_multiplier", 1.0)),
    }


def _log_retrieval_debug(
    claim: str,
    filters: list[tuple[str, str]],
    articles: list,
    stage: str = "retrieval",
) -> None:
    """Print retrieval debug lines for each returned article."""
    print(f"\n{'─'*70}")
    print(f"🔬 RETRIEVAL DEBUG [{stage}]")
    print(f"{'─'*70}")
    print(f"Claim: {claim[:180]}{'...' if len(claim) > 180 else ''}")
    print(
        "Filtri: "
        + ", ".join(f"{source}/{libro or 'N/A'}" for source, libro in filters if source)
    )
    if not articles:
        print("⚠️ Nessun articolo recuperato.")
        return
    print("Top articoli (con breakdown score):")
    for i, art in enumerate(articles, start=1):
        payload = _article_with_retrieval_debug(art)
        print(
            f"{i:02d}. [{payload['source']}] Art. {payload['articolo']} "
            f"| score={payload['score']:.4f} "
            f"| v_rank={payload['vector_rank_score']:.4f} "
            f"| ft_rank={payload['fulltext_rank_score']:.4f} "
            f"| fusion={payload['fusion_score']:.4f} "
            f"| kw={payload['keyword_bonus']:.4f} "
            f"| priority={payload['priority_multiplier']:.4f}"
        )


def prepare_claim_context(
    claim: str,
    include_precedents: bool,
    max_statutes: int,
    max_precedents: int,
    progress_callback=None,
) -> tuple[list[dict], list[dict]]:
    """Pre-retrieve statutes and precedents before reasoning.

    Implements progressive search: if the initial top_k search yields fewer
    than `search_min_kept_statutes` after relevance filtering, the search
    expands by `search_expansion_step` at a time (up to `search_max_expansions`
    rounds) until the minimum is met or no more articles can be found.
    """
    progress_callback = progress_callback or (lambda _detail, _progress: None)

    min_kept = settings.search_min_kept_statutes
    expansion_step = settings.search_expansion_step
    max_expansions = settings.search_max_expansions

    print(
        f"🔎 Pre-retrieval config: top_k_statutes={max_statutes}, "
        f"min_kept={min_kept}, max_precedents={max_precedents}"
    )

    pipe = get_pipeline()
    reas = get_reasoner()

    # ── Step 1: classify + embed once ──────────────────────────────────
    progress_callback("Classificazione claim e embedding query", 18)
    classification = pipe.classifier.classify(claim)
    embedding = pipe.embed_text(claim)
    libri_filters = pipe.build_search_filters(
        classification, settings.search_use_top_n_libri
    )

    # ── Step 2: initial hybrid retrieval ───────────────────────────────
    progress_callback("Recupero norme candidate (vettoriale + fulltext)", 28)
    current_top_k = max_statutes
    articles = pipe.vector_search(
        embedding,
        libri_filters,
        current_top_k,
        query_text=claim,
    )
    _log_retrieval_debug(
        claim=claim,
        filters=libri_filters,
        articles=articles,
        stage=f"initial_top_k_{current_top_k}",
    )
    statutes = _articles_to_dicts(articles)
    legal_context = reas._extract_legal_context(claim)
    progress_callback("Filtro rilevanza norme", 38)
    kept_statutes = reas.filter_irrelevant_statutes(claim, statutes)
    progress_callback("Verifica applicabilità norme", 48)
    kept_statutes = reas.filter_applicable_statutes(claim, kept_statutes, legal_context)
    seen_ids = {s["statute_id"] for s in statutes}  # all fetched so far

    print(
        f"📊 Initial search: {len(statutes)} fetched, "
        f"{len(kept_statutes)} kept (min={min_kept})"
    )

    # ── Step 3: progressive expansion ──────────────────────────────────
    expansion = 0
    while len(kept_statutes) < min_kept and expansion < max_expansions:
        expansion += 1
        current_top_k += expansion_step
        progress_callback(
            f"Espansione recupero norme ({expansion}/{max_expansions})",
            min(58, 50 + expansion * 2),
        )
        print(
            f"🔄 Expansion {expansion}/{max_expansions}: "
            f"top_k={current_top_k}, kept so far={len(kept_statutes)}"
        )

        # Re-query with larger top_k (embedding & classification reused)
        articles = pipe.vector_search(
            embedding,
            libri_filters,
            current_top_k,
            query_text=claim,
        )
        _log_retrieval_debug(
            claim=claim,
            filters=libri_filters,
            articles=articles,
            stage=f"expansion_{expansion}_top_k_{current_top_k}",
        )
        new_statutes = [
            d for d in _articles_to_dicts(articles) if d["statute_id"] not in seen_ids
        ]

        if not new_statutes:
            print("   ⚠️ No new articles found — stopping expansion")
            break

        seen_ids.update(s["statute_id"] for s in new_statutes)
        progress_callback("Filtro rilevanza norme (espansione)", 60)
        new_kept = reas.filter_irrelevant_statutes(claim, new_statutes)
        progress_callback("Verifica applicabilità norme (espansione)", 62)
        new_kept = reas.filter_applicable_statutes(claim, new_kept, legal_context)
        kept_statutes.extend(new_kept)

        print(
            f"   📊 +{len(new_statutes)} new fetched, "
            f"+{len(new_kept)} kept → total kept={len(kept_statutes)}"
        )

    if expansion > 0:
        print(
            f"✅ Progressive search complete: {expansion} expansion(s), "
            f"{len(kept_statutes)} statutes kept"
        )

    statutes = kept_statutes

    # ── Precedents (unchanged) ─────────────────────────────────────────
    precedents: list[dict] = []
    if include_precedents:
        progress_callback("Recupero precedenti", 64)
        try:
            result = search_precedents_tool.invoke(
                {"query": claim, "limit": max_precedents}
            )
            if isinstance(result, list):
                precedents = result
        except Exception as e:
            print(f"⚠️ Errore recupero precedenti: {e}")

    progress_callback("Filtro precedenti", 66)
    precedents = reas.filter_irrelevant_precedents(claim, precedents)

    return statutes, precedents


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
            additional_causal_types=[],
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


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {"status": "ok", "service": "LexCausa API", "version": settings.api_version}
    )


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """
    Return current default settings and available models.

    The frontend uses this to populate the Settings panel.
    """
    selectable_models = settings.available_model_aliases

    return jsonify(
        {
            "models": selectable_models,
            "model_mapping": settings.model_alias_map,
            "defaults": {
                "reasoner_model": settings.reasoner_default_model,
                "counter_model": settings.counter_default_model,
                "pipeline_model_order": settings.pipeline_model_order_aliases,
                "llm_temperature": settings.llm_temperature,
                "llm_max_tokens": settings.llm_max_tokens,
                "search_top_k_default": settings.search_top_k_default,
                "search_min_kept_statutes": settings.search_min_kept_statutes,
                "search_use_top_n_libri": settings.search_use_top_n_libri,
                "precedents_limit_default": settings.precedents_limit_default,
                "include_precedents": True,
                "chain_min_steps": settings.chain_min_steps,
                "chain_max_steps": settings.chain_max_steps,
                "aqa_alpha": settings.aqa_alpha,
                "aqa_beta": settings.aqa_beta,
                "aqa_gamma": settings.aqa_gamma,
                "aqa_min_semantic_overlap": settings.aqa_min_semantic_overlap,
                "aqa_min_strength_ratio": settings.aqa_min_strength_ratio,
                "aqa_damage_factor": settings.aqa_damage_factor,
                "aqa_allow_factual_attacks": settings.aqa_allow_factual_attacks,
                "aqa_allow_cross_codice": settings.aqa_allow_cross_codice,
                "aqa_strength_ratio_by_type": settings.aqa_strength_ratio_by_type,
                "enable_causality": True,
            },
        }
    )


def _build_agent_config(
    model_override: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AgentConfig:
    """Build an AgentConfig from optional frontend overrides."""
    return AgentConfig(
        model_name=settings.resolve_model_name(model_override),
        temperature=(
            temperature if temperature is not None else settings.llm_temperature
        ),
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
    )


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


def _sse_event(event: str, payload: dict) -> str:
    """Encode one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_full_pipeline(
    data: dict,
    *,
    status_callback=None,
    token_callback=None,
    progress_callback=None,
) -> dict:
    """Execute the full pipeline and optionally emit status/token callbacks."""
    if not isinstance(data, dict):
        data = {}

    status_callback = status_callback or (lambda _msg: None)
    token_callback = token_callback or (lambda _payload: None)
    progress_callback = progress_callback or (lambda _event, _payload: None)

    def _emit_progress(event_name: str, payload: dict) -> None:
        try:
            progress_callback(event_name, payload)
        except Exception:
            # Progress streaming must never break pipeline execution.
            pass

    def _emit_phase(
        phase: str, status: str, progress: int, detail: str | None = None
    ) -> None:
        _emit_progress(
            "phase",
            {
                "phase": phase,
                "status": status,
                "progress": max(0, min(100, int(progress))),
                "detail": detail or "",
            },
        )

    claim = (data.get("claim", "") or "").strip()
    include_precedents = data.get("include_precedents", True)
    max_statutes = data.get("max_statutes", settings.search_top_k_default)
    max_precedents = data.get("max_precedents", settings.precedents_limit_default)

    # ── Frontend-configurable settings ────────────────────────────────
    fe_settings = data.get("settings", {}) or {}
    fe_temperature = fe_settings.get("llm_temperature")
    fe_max_tokens = fe_settings.get("llm_max_tokens")
    # Per-step model selection (alias resolved via settings model map)
    fe_reasoner_model = fe_settings.get("reasoner_model")
    fe_counter_model = fe_settings.get("counter_model")
    # AQA weights
    fe_aqa_alpha = fe_settings.get("aqa_alpha")
    fe_aqa_beta = fe_settings.get("aqa_beta")
    fe_aqa_gamma = fe_settings.get("aqa_gamma")
    # AQA attack parameters
    fe_aqa_min_semantic_overlap = fe_settings.get("aqa_min_semantic_overlap")
    fe_aqa_min_strength_ratio = fe_settings.get("aqa_min_strength_ratio")
    fe_aqa_damage_factor = fe_settings.get("aqa_damage_factor")
    fe_aqa_allow_factual_attacks = fe_settings.get("aqa_allow_factual_attacks")
    fe_aqa_allow_cross_codice = fe_settings.get("aqa_allow_cross_codice")
    fe_aqa_strength_ratio_by_type = fe_settings.get("aqa_strength_ratio_by_type")
    fe_search_min_kept_statutes = fe_settings.get("search_min_kept_statutes")
    fe_chain_min_steps = fe_settings.get("chain_min_steps")
    fe_chain_max_steps = fe_settings.get("chain_max_steps")
    fe_enable_causality = fe_settings.get("enable_causality", True)

    if not claim:
        raise ValueError('Campo "claim" obbligatorio')

    status_callback("Avvio pipeline completa...")
    _emit_phase("context_setup", "active", 5, "Routing iniziale")

    with _pipeline_logger(claim):
        print(f"\n{'='*70}")
        print("🚀 FULL PIPELINE - START")
        print(f"{'='*70}")
        print(f"Claim: {claim[:100]}...")

        if fe_settings:
            print(f"⚙️  Frontend settings override: {fe_settings}")
        if not fe_enable_causality:
            print("🔬 Causality taxonomy DISABLED by frontend settings")

        # Apply chain step overrides to global settings
        if fe_chain_min_steps is not None:
            settings.chain_min_steps = int(fe_chain_min_steps)
        if fe_chain_max_steps is not None:
            settings.chain_max_steps = int(fe_chain_max_steps)
        if fe_search_min_kept_statutes is not None:
            settings.search_min_kept_statutes = int(fe_search_min_kept_statutes)

        status_callback("Preparazione contesto giuridico...")
        _emit_phase("context_setup", "active", 15, "Routing dominio e causalità")
        routing_decision = resolve_routing_decision(claim, data)
        _emit_phase("context_setup", "active", 20, "Avvio recupero contesto")

        # Preload context once for both reasoners
        def _emit_context_detail(detail: str, progress: int) -> None:
            _emit_phase("context_setup", "active", progress, detail)

        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            progress_callback=_emit_context_detail,
        )
        _emit_phase(
            "context_setup",
            "active",
            68,
            "Classificazione stance NLI (norme e precedenti)",
        )

        # Classify stance using NLI to separate support vs against
        (
            support_statutes,
            against_statutes,
            support_precedents,
            against_precedents,
        ) = classify_stance_for_agents(claim, statutes, precedents)

        # STEP 1: main supportive reasoning
        print(f"\n{'─'*70}")
        print("📊 STEP 1: Reasoner execution (SUPPORT articles)...")
        print(f"{'─'*70}")
        print(
            f"   📚 Knowledge base: {len(support_statutes)} statutes, {len(support_precedents)} precedents"
        )
        _emit_phase("context_setup", "done", 100, "Contesto pronto")
        _emit_phase("support", "active", 8, "Generazione ragionamento")
        status_callback("Generazione argomentazione principale in corso...")

        reasoner_config = _build_agent_config(
            model_override=fe_reasoner_model,
            temperature=fe_temperature,
            max_tokens=fe_max_tokens,
        )
        reas = Reasoner(config=reasoner_config)
        reasoner_result = reas.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=support_statutes,
            pre_retrieved_precedents=support_precedents,
            enable_causality=fe_enable_causality,
            stream_callback=token_callback,
        )
        _emit_phase("support", "active", 97, "Costruzione ASPIC+ e output finale")
        _emit_progress("reasoner_result", reasoner_result.to_dict())
        final_routing_decision = RoutingDecision(
            claim=claim,
            domain=routing_decision.domain,
            causal_type_id=reasoner_result.causal_type_id,
            theory_id=reasoner_result.theory_id,
            anchor_norms=reasoner_result.anchor_norms,
            principle_tests=reasoner_result.principle_tests,
            additional_causal_types=reasoner_result.causal_type_ids_for_counter,
        )

        print("✅ Reasoner completed")
        print(
            f"   - Domain: {routing_decision.domain} -> Causal type: {final_routing_decision.causal_type_id} / {final_routing_decision.theory_id}"
        )
        print(f"   - Mismatch status: {reasoner_result.mismatch_status}")
        print(
            f"   - Anchor norms: core={len(reasoner_result.anchor_norms.get('core_norms', []))}, accessory={len(reasoner_result.anchor_norms.get('accessory_norms', []))}"
        )
        print(f"   - Statutes for reasoning: {len(reasoner_result.relevant_statutes)}")
        print(f"   - Arguments: {len(reasoner_result.arguments)}")
        print(f"   - Reasoning chain: {len(reasoner_result.reasoning_chain)} steps")
        _emit_phase("support", "done", 100, "Argomentazione completata")
        _emit_phase("counter", "active", 8, "Generazione contro-argomentazione")

        # STEP 2: counter reasoning
        print(f"\n{'─'*70}")
        print("⚔️  STEP 2: Counter-Reasoner execution (AGAINST articles)...")
        print(f"{'─'*70}")
        print(
            f"   📚 Knowledge base: {len(against_statutes)} statutes, {len(against_precedents)} precedents"
        )
        status_callback("Generazione argomentazione contraria in corso...")

        counter_config = _build_agent_config(
            model_override=fe_counter_model,
            temperature=fe_temperature,
            max_tokens=fe_max_tokens,
        )
        cr = CounterReasoner(config=counter_config)

        # Opposition consistency is validated by the Polisher gate.
        reasoner_conclusion = reasoner_result.conclusion or ""
        if not reasoner_conclusion:
            print(
                "ℹ️ Reasoner conclusion unavailable: no fallback applied. "
                "Opposition check is delegated to Polisher gate."
            )
        else:
            print(
                "ℹ️ Reasoner conclusion captured for Counter traceability: "
                f"{reasoner_conclusion[:120]}..."
            )

        counter_result = cr.run(
            claim=claim,
            routing_decision=final_routing_decision,
            pre_retrieved_statutes=against_statutes,
            pre_retrieved_precedents=against_precedents,
            enable_causality=fe_enable_causality,
            reasoner_conclusion=reasoner_conclusion,
            stream_callback=token_callback,
        )
        _emit_phase("counter", "active", 97, "Costruzione ASPIC+ e output finale")
        counter_result.reasoner_conclusion_context = reasoner_conclusion

        print("✅ Counter-Reasoner completed")
        print(
            f"   - Causal type: {counter_result.causal_type_id} / {counter_result.theory_id}"
        )
        print(
            "   - Selected attacks: "
            + (
                ", ".join(counter_result.selected_attack_ids)
                if counter_result.selected_attack_ids
                else counter_result.selected_attack_id
            )
        )
        print(
            f"   - Statutes for counter-reasoning: {len(counter_result.relevant_statutes)}"
        )
        print(f"   - Counter-arguments: {len(counter_result.counter_arguments)}")
        print(
            f"   - Counter-reasoning chain: {len(counter_result.reasoning_chain)} steps"
        )
        # Emit the complete counter output immediately so the frontend can render
        # the same structured view as the reasoner before evaluation starts.
        _emit_progress("counter_result", counter_result.to_dict())
        _emit_phase("counter", "done", 100, "Contro-argomentazione completata")
        _emit_phase("final_evaluation", "active", 10, "Avvio verifica consistenza")

        # STEP 3: final consistency/evaluation
        print(f"\n{'─'*70}")
        print("📊 STEP 3: Polisher-Evaluator (consistency check)...")
        print(f"{'─'*70}")
        status_callback("Verifica finale e valutazione in corso...")

        pe = get_polisher_evaluator()

        # Apply AQA weight overrides if provided by frontend
        if fe_aqa_alpha is not None:
            settings.aqa_alpha = float(fe_aqa_alpha)
        if fe_aqa_beta is not None:
            settings.aqa_beta = float(fe_aqa_beta)
        if fe_aqa_gamma is not None:
            settings.aqa_gamma = float(fe_aqa_gamma)
        # Apply AQA attack parameter overrides
        if fe_aqa_min_semantic_overlap is not None:
            settings.aqa_min_semantic_overlap = float(fe_aqa_min_semantic_overlap)
        if fe_aqa_min_strength_ratio is not None:
            settings.aqa_min_strength_ratio = float(fe_aqa_min_strength_ratio)
        if fe_aqa_damage_factor is not None:
            settings.aqa_damage_factor = float(fe_aqa_damage_factor)
        if fe_aqa_allow_factual_attacks is not None:
            settings.aqa_allow_factual_attacks = bool(fe_aqa_allow_factual_attacks)
        if fe_aqa_allow_cross_codice is not None:
            settings.aqa_allow_cross_codice = bool(fe_aqa_allow_cross_codice)
        if fe_aqa_strength_ratio_by_type is not None:
            settings.aqa_strength_ratio_by_type = fe_aqa_strength_ratio_by_type

        evaluation_result = pe.run(
            claim=claim,
            domain=final_routing_decision.domain,
            reasoner_output=reasoner_result.to_dict(),
            counter_reasoner_output=counter_result.to_dict(),
            progress_callback=_emit_progress,
        )

        # Apply Polisher counter gate to the returned counter output
        counter_gate = (
            (evaluation_result.counter_reasoner_gate or {})
            if hasattr(evaluation_result, "counter_reasoner_gate")
            else {}
        )
        if counter_gate.get("abstain"):
            counter_result.abstained = True
            if not counter_result.abstention_reason:
                counter_result.abstention_reason = counter_gate.get(
                    "reason",
                    "Il Counter-Reasoner non ha abbastanza materiale per argomentare contro.",
                )
            print("⚠️ Counter-Reasoner aggiornato dal Polisher gate")
            print(f"   - Gate label: {counter_gate.get('label')}")
            print(
                f"   - Gate reason: {counter_gate.get('reason', counter_result.abstention_reason)}"
            )
        # Re-emit after evaluator in case the counter gate updated abstention fields.
        _emit_progress("counter_result", counter_result.to_dict())

        # Derive winning_side and confidence from AQA verdict
        aqa = evaluation_result.aqa_report or {}
        aqa_verdict = aqa.get("verdict", "uncertain")
        aqa_net = aqa.get("net_plausibility", {})
        verdict_map = {
            "plausible": "support",
            "implausible": "counter",
            "uncertain": "undecided",
        }
        evaluation_result.winning_side = verdict_map.get(aqa_verdict, "undecided")
        evaluation_result.confidence = abs(aqa_net.get("final", 0.0))
        evaluation_payload = evaluation_result.to_dict()
        repaired_aspic_files = _persist_repaired_aspic_files(claim, evaluation_payload)
        if repaired_aspic_files:
            evaluation_payload["repaired_aspic_files"] = repaired_aspic_files
            _emit_progress(
                "evaluation_partial",
                {"repaired_aspic_files": repaired_aspic_files},
            )
        aqa_report_file = _persist_aqa_report_file(claim, evaluation_payload)
        if aqa_report_file:
            evaluation_payload["aqa_report_file"] = aqa_report_file
            _emit_progress("evaluation_partial", {"aqa_report_file": aqa_report_file})
        _emit_progress("evaluation_result", evaluation_payload)

        print("✅ Polisher-Evaluator completed")
        print(f"   - Winning side: {evaluation_result.winning_side}")
        print(f"   - Confidence: {evaluation_result.confidence:.2f}")
        print(
            f"   - AQA verdict: {aqa_verdict} "
            f"(pro={aqa_net.get('pro', 0):.2f}, "
            f"contra={aqa_net.get('contra', 0):.2f}, "
            f"final={aqa_net.get('final', 0):.2f})"
        )

        print(f"\n{'='*70}")
        print("✅ FULL PIPELINE - END")
        print(f"{'='*70}\n")

    status_callback("Pipeline completata.")
    _emit_phase("final_evaluation", "done", 100, "Valutazione completata")
    return {
        "claim": claim,
        "routing": routing_decision.to_dict(),
        "final_routing": final_routing_decision.to_dict(),
        "reasoner": reasoner_result.to_dict(),
        "counter_reasoner": counter_result.to_dict(),
        "evaluation": evaluation_payload,
    }


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    """
    Endpoint JSON classico per la pipeline completa.
    """
    if not _pipeline_lock.acquire(blocking=False):
        return jsonify({"error": "A pipeline is already running. Please wait."}), 429

    try:
        data = request.get_json(silent=True) or {}
        result = _run_full_pipeline(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ PIPELINE ERROR")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        _pipeline_lock.release()


@app.route("/api/pipeline/stream", methods=["POST"])
def pipeline_stream():
    """
    Endpoint SSE con streaming token-by-token della pipeline completa.
    """
    if not _pipeline_lock.acquire(blocking=False):
        return jsonify({"error": "A pipeline is already running. Please wait."}), 429

    data = request.get_json(silent=True) or {}
    event_queue: Queue = Queue()
    sentinel = object()

    def push_status(message: str) -> None:
        event_queue.put(("status", {"message": message}))

    def push_token(payload: dict) -> None:
        event_queue.put(("token", payload))

    def push_progress(event_name: str, payload: dict) -> None:
        event_queue.put((event_name, payload))

    def push_error(message: str, code: int) -> None:
        event_queue.put(("error", {"message": message, "code": code}))

    def _worker() -> None:
        try:
            result = _run_full_pipeline(
                data,
                status_callback=push_status,
                token_callback=push_token,
                progress_callback=push_progress,
            )
            event_queue.put(("final", result))
        except ValueError as e:
            push_error(str(e), 400)
        except Exception as e:
            print(f"\n{'='*70}")
            print("❌ PIPELINE STREAM ERROR")
            print(f"{'='*70}")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            push_error(str(e), 500)
        finally:
            event_queue.put(("done", {"ok": True}))
            event_queue.put(sentinel)
            _pipeline_lock.release()

    threading.Thread(target=_worker, daemon=True).start()

    @stream_with_context
    def generate():
        while True:
            try:
                item = event_queue.get(timeout=12)
            except Empty:
                # Keep proxies/tunnels alive during long LLM phases with no tokens.
                yield _sse_event("heartbeat", {"ts": int(time.time())})
                continue
            if item is sentinel:
                break
            event, payload = item
            yield _sse_event(event, payload)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Endpoint per la valutazione finale.
    Verifica la consistenza delle argomentazioni con la knowledge base via Neo4j.
    """
    try:
        data = request.get_json()
        claim = data.get("claim", "").strip()
        domain = data.get("domain", "ENTRAMBI")
        reasoner_output = data.get("reasoner_output", {})
        counter_output = data.get("counter_output", {})

        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        pe = get_polisher_evaluator()
        result = pe.run(
            claim=claim,
            domain=domain,
            reasoner_output=reasoner_output,
            counter_reasoner_output=counter_output,
        )

        payload = result.to_dict()
        repaired_aspic_files = _persist_repaired_aspic_files(claim, payload)
        if repaired_aspic_files:
            payload["repaired_aspic_files"] = repaired_aspic_files
        aqa_report_file = _persist_aqa_report_file(claim, payload)
        if aqa_report_file:
            payload["aqa_report_file"] = aqa_report_file
        return jsonify(payload)

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
        if source == "codice_civile":
            source_label = "Codice Civile"
            src = "CC"
        elif source == "codice_penale":
            source_label = "Codice Penale"
            src = "CP"
        elif source == "codice_amministrativo":
            source_label = "Codice Amministrativo (L. 241/1990)"
            src = "AMM"
        else:
            source_label = source or "Codice"
            src = "COD"

        lines.append(f"**{i}. {cat}**")
        lines.append(f"- {desc}")
        if libro:
            lines.append(f"- 📚 [{source_label}] {libro}\n")
        else:
            lines.append(f"- 📚 [{source_label}] (intero codice, senza libri)\n")

    if result.articles:
        lines.append("\n---\n")
        lines.append("## 📖 Articoli Rilevanti\n")
        for i, art in enumerate(result.articles, 1):
            if art.source == "codice_civile":
                src = "CC"
            elif art.source == "codice_penale":
                src = "CP"
            elif art.source == "codice_amministrativo":
                src = "AMM"
            else:
                src = "COD"
            lines.append(f"### {i}. [{src}] Art. {art.articolo} - {art.titolo}")
            lines.append(f"**Rilevanza:** {art.score:.1%}")
            if art.libro:
                lines.append(f"**Libro:** {art.libro}\n")
            else:
                lines.append("**Libro:** N/A (codice senza libri)\n")
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
    print("  • GET  /api/settings        - Impostazioni configurabili")
    print("  • POST /api/chat            - Ricerca legale (Tab Ricerca)")
    print("  • POST /api/reason          - Ragionamento causale (Tab Ragionamento)")
    print("  • POST /api/counter_reason  - Contro-ragionamento")
    print("  • POST /api/pipeline        - Pipeline completa")
    print("  • POST /api/pipeline/stream - Pipeline completa (SSE token streaming)")
    print("  • POST /api/evaluate        - Valutazione finale (stub)")
    print()
    print("=" * 70)
    print()

    app.run(host=settings.api_host, port=settings.api_port, debug=settings.debug)
