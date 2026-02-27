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
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any

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

from agents import (  # noqa: E402
    CounterReasoner,
    PolisherEvaluator,
    Reasoner,
    RetrievalFilterAgent,
)
from agents.base import AgentConfig  # noqa: E402
from agents.base import retrieval_llm_fail_fast_scope  # noqa: E402
from agents.router import Router, RoutingDecision  # noqa: E402
from agents.tools import config_loader  # noqa: E402
from agents.tools.neo4j_tools import (  # noqa: E402
    get_legal_search_pipeline,
    search_precedents_tool,
)
from config import settings  # noqa: E402
from services.claim_context_memory import get_claim_context_memory  # noqa: E402
from services.pipeline_control import PipelineCancelled  # noqa: E402

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Prevent concurrent pipeline executions (interleaves logs & shared state)
_pipeline_lock = threading.Lock()
_active_stream_run_lock = threading.Lock()
_active_stream_run: dict | None = None
_retrieval_model_override_lock = threading.Lock()

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


def _artifact_file_payload(path: Path) -> dict[str, str]:
    """Build standard absolute/relative file payload."""
    return {
        "absolute_path": str(path),
        "relative_path": str(path.relative_to(project_root)),
    }


def _render_doe_log_section(section_name: str, payload: dict) -> str:
    """Render one human-readable DoE section with summary + raw artifacts."""
    entry: dict[str, Any] = payload if isinstance(payload, dict) else {}
    view_raw = entry.get("view")
    view: dict[str, Any] = view_raw if isinstance(view_raw, dict) else {}
    counter_raw = view.get("counter_reasoner")
    counter: dict[str, Any] = counter_raw if isinstance(counter_raw, dict) else {}
    evaluation_raw = view.get("evaluation")
    evaluation: dict[str, Any] = (
        evaluation_raw if isinstance(evaluation_raw, dict) else {}
    )
    lines = [f"[{section_name}]"]
    lines.append(f"label: {entry.get('label') or section_name}")
    lines.append(f"description: {entry.get('description') or '-'}")
    lines.append(f"status: {entry.get('status') or '-'}")
    lines.append(f"duration_ms: {entry.get('duration_ms')}")
    lines.append("[SETTINGS]")
    lines.append(json.dumps(entry.get("settings") or {}, ensure_ascii=False, indent=2))
    lines.append("[METRICS]")
    lines.append(json.dumps(entry.get("metrics") or {}, ensure_ascii=False, indent=2))

    if counter:
        counter_meta = {
            "selected_attack_id": counter.get("selected_attack_id"),
            "selected_attack_ids": counter.get("selected_attack_ids") or [],
            "reasoner_causality": counter.get("reasoner_causality") or {},
            "abstained": bool(counter.get("abstained")),
            "statutes_count": len(counter.get("statutes") or []),
            "precedents_count": len(counter.get("precedents") or []),
        }
        lines.append("[COUNTER_METADATA]")
        lines.append(json.dumps(counter_meta, ensure_ascii=False, indent=2))
        if counter.get("raw_response"):
            lines.append("[COUNTER_RESPONSE]")
            lines.append(str(counter.get("raw_response") or "").strip())

    if evaluation:
        if evaluation.get("summary"):
            lines.append("[EVALUATION_SUMMARY]")
            lines.append(str(evaluation.get("summary") or "").strip())
        if evaluation.get("aqa_report"):
            lines.append("[AQA_REPORT]")
            lines.append(
                json.dumps(
                    evaluation.get("aqa_report") or {},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if evaluation.get("repaired_aspic_files"):
            lines.append("[REPAIRED_ASPIC_FILES]")
            lines.append(
                json.dumps(
                    evaluation.get("repaired_aspic_files") or {},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if evaluation.get("aqa_report_file"):
            lines.append("[AQA_REPORT_FILE]")
            lines.append(
                json.dumps(
                    evaluation.get("aqa_report_file") or {},
                    ensure_ascii=False,
                    indent=2,
                )
            )

    return "\n".join(lines).strip()


def _persist_doe_experiment_files(claim: str, doe_payload: dict) -> dict:
    """Persist one consolidated DoE log + JSON report with explicit A/B sections."""
    if not isinstance(doe_payload, dict):
        return {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify_filename(claim)
    report_dir = LOG_DIR / "doe_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{ts}_{slug}_doe.log"
    report_path = report_dir / f"{ts}_{slug}_doe_report.json"

    reasoner_shared = (
        doe_payload.get("reasoner_shared")
        if isinstance(doe_payload.get("reasoner_shared"), dict)
        else {}
    )
    baseline_raw = doe_payload.get("baseline")
    baseline: dict[str, Any] = baseline_raw if isinstance(baseline_raw, dict) else {}
    treatment_raw = doe_payload.get("treatment")
    treatment: dict[str, Any] = treatment_raw if isinstance(treatment_raw, dict) else {}
    delta = (
        doe_payload.get("delta") if isinstance(doe_payload.get("delta"), dict) else {}
    )

    log_sections = [
        f"[{datetime.now().isoformat()}] DoE log for claim:",
        claim,
        "=" * 70,
        "",
        "[DOE_META]",
        json.dumps(
            {
                "mode": doe_payload.get("mode") or "automatic_ab",
                "generated_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "[SHARED_REASONER]",
        json.dumps(reasoner_shared, ensure_ascii=False, indent=2),
        "",
        _render_doe_log_section("DOE-A", baseline),
        "",
        _render_doe_log_section("DOE-B", treatment),
        "",
        "[DOE-DELTA]",
        json.dumps(delta, ensure_ascii=False, indent=2),
        "",
    ]

    try:
        log_path.write_text("\n".join(log_sections), encoding="utf-8")
        report_path.write_text(
            json.dumps(doe_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"⚠️ Errore salvataggio report/log DoE: {exc}")
        return {}

    artifacts = {
        "doe_log_file": _artifact_file_payload(log_path),
        "doe_report_file": _artifact_file_payload(report_path),
    }
    print(f"🧪 DoE consolidato salvato: {artifacts}")
    return artifacts


def _persist_pdf_export_file(
    claim: str,
    pdf_bytes: bytes,
    *,
    export_context: str = "pipeline",
    prefix: str = "pipeline",
    client_filename: str = "",
) -> dict:
    """Persist one exported PDF under logs/pdf_exports/<context>/."""
    if not pdf_bytes:
        raise ValueError("Contenuto PDF mancante")

    normalized_context = (export_context or "pipeline").strip().lower()
    if normalized_context not in {"pipeline", "doe"}:
        normalized_context = "pipeline"

    out_dir = LOG_DIR / "pdf_exports" / normalized_context
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify_filename(claim or client_filename or "claim")
    prefix_slug = _slugify_filename(prefix or normalized_context, max_len=40)
    filename = f"{ts}_{slug}_{prefix_slug}.pdf"
    path = out_dir / filename

    try:
        path.write_bytes(pdf_bytes)
    except Exception as exc:
        print(f"⚠️ Errore salvataggio PDF export: {exc}")
        return {}

    payload = _artifact_file_payload(path)
    print(f"📄 PDF export salvato: {payload}")
    return payload


# Agenti globali (lazy initialization)
reasoner = None
counter_reasoner = None
polisher_evaluator = None
router_agent = None
retrieval_filter_agent = None

# Non-reasoning pipeline helpers must be deterministic and not affected by the
# frontend temperature slider. Reasoner/CounterReasoner receive their own
# per-request temperature explicitly.
PIPELINE_AUX_LLM_TEMPERATURE = 0.0


def get_pipeline():
    """Get the shared LegalSearchPipeline singleton."""
    return get_legal_search_pipeline()


def get_retrieval_filter_agent():
    """Lazy load helper agent for retrieval filtering (not Reasoner)."""
    global retrieval_filter_agent
    if retrieval_filter_agent is None:
        print("🔧 Inizializzazione Motore Retrieval...")
        retrieval_filter_agent = RetrievalFilterAgent(
            config=AgentConfig(
                model_name=settings.retrieval_default_model,
                temperature=PIPELINE_AUX_LLM_TEMPERATURE,
                max_tokens=settings.llm_max_tokens,
            )
        )
        print("✅ Motore Retrieval pronto!")
    return retrieval_filter_agent


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
            "score": float(getattr(art, "score", 0.0)),
        }
        for art in articles
    ]


def _log_claim_context_items(
    *,
    claim: str,
    statutes: list[dict],
    precedents: list[dict],
    source_label: str,
) -> None:
    """Print a compact summary of claim-context items (e.g., cache HIT contents)."""
    print(f"💾 [Retrieval] Claim-context items ({source_label}) for claim:")
    print(f"   - {claim[:160]}{'...' if len(claim) > 160 else ''}")
    if statutes:
        print(f"   - Statutes ({len(statutes)}):")
        for idx, art in enumerate(statutes, start=1):
            articolo = art.get("articolo") or art.get("statute_id") or "?"
            titolo = art.get("titolo") or ""
            source = art.get("source") or ""
            source_suffix = f" ({source})" if source else ""
            title_suffix = f" - {titolo}" if titolo else ""
            print(f"     [{idx}] Art. {articolo}{source_suffix}{title_suffix}")
    else:
        print("   - Statutes: none")

    if precedents:
        print(f"   - Precedents ({len(precedents)}):")
        for idx, pr in enumerate(precedents, start=1):
            title = (
                pr.get("title")
                or pr.get("titolo")
                or pr.get("name")
                or f"Precedent {idx}"
            )
            print(f"     [{idx}] {title}")
    else:
        print("   - Precedents: none")


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
    top_n = int(settings.search_retrieval_debug_top_n)
    if top_n <= 0:
        return
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
    shown = articles[:top_n]
    print(f"Top articoli (con breakdown score) — showing {len(shown)}/{len(articles)}:")
    for i, art in enumerate(shown, start=1):
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
    if len(articles) > top_n:
        print(
            f"... {len(articles) - top_n} articoli ulteriori omessi (config: SEARCH_RETRIEVAL_DEBUG_TOP_N={top_n})"
        )


def _build_claim_context_memory_signature() -> dict:
    """Versioned signature for pre-retrieval cache invalidation."""
    return {
        "algo_version": "pre_retrieval_claim_context_v1",
        "search_min_kept_statutes": int(settings.search_min_kept_statutes),
        "search_expansion_step": int(settings.search_expansion_step),
        "search_max_expansions": int(settings.search_max_expansions),
        "search_expansion_max_zero_gain_rounds": int(
            settings.search_expansion_max_zero_gain_rounds
        ),
        "search_use_top_n_libri": int(settings.search_use_top_n_libri),
        "search_query_terms_mode": str(settings.search_query_terms_mode),
        "search_query_terms_llm_max_terms": int(
            settings.search_query_terms_llm_max_terms
        ),
        "search_query_terms_llm_max_tokens": int(
            settings.search_query_terms_llm_max_tokens
        ),
        "search_cites_per_article_limit": int(settings.search_cites_per_article_limit),
        "search_cites_max_additional": int(settings.search_cites_max_additional),
        "search_cites_score_decay": float(settings.search_cites_score_decay),
        "search_hybrid_penale_vector_weight": float(
            settings.search_hybrid_penale_vector_weight
        ),
        "search_hybrid_penale_fulltext_weight": float(
            settings.search_hybrid_penale_fulltext_weight
        ),
        "search_hybrid_civile_vector_weight": float(
            settings.search_hybrid_civile_vector_weight
        ),
        "search_hybrid_civile_fulltext_weight": float(
            settings.search_hybrid_civile_fulltext_weight
        ),
        "search_hybrid_admin_vector_weight": float(
            settings.search_hybrid_admin_vector_weight
        ),
        "search_hybrid_admin_fulltext_weight": float(
            settings.search_hybrid_admin_fulltext_weight
        ),
    }


def prepare_claim_context(
    claim: str,
    include_precedents: bool,
    max_statutes: int,
    max_precedents: int,
    claim_context_memory_enabled: bool = False,
    claim_context_memory_overwrite: bool = False,
    progress_callback=None,
    cancel_checker=None,
    result_metadata: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Pre-retrieve statutes and precedents before reasoning.

    Implements progressive search: if the initial top_k search yields fewer
    than `search_min_kept_statutes` after relevance filtering, the search
    expands by `search_expansion_step` at a time (up to `search_max_expansions`
    rounds) until the minimum is met or no more articles can be found.
    """
    progress_callback = progress_callback or (lambda _detail, _progress: None)
    cancel_checker = cancel_checker or (lambda: False)

    def _check_cancel() -> None:
        if cancel_checker():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")

    min_kept = settings.search_min_kept_statutes
    expansion_step = settings.search_expansion_step
    max_expansions = settings.search_max_expansions
    memory_enabled = bool(claim_context_memory_enabled)
    memory_overwrite = bool(claim_context_memory_overwrite)
    if memory_overwrite and not memory_enabled:
        memory_enabled = True
    if isinstance(result_metadata, dict):
        result_metadata.clear()
        result_metadata["memory_enabled"] = memory_enabled
        result_metadata["memory_overwrite"] = memory_overwrite
        result_metadata["memory_hit"] = False

    print(
        f"🔎 Pre-retrieval config: top_k_statutes={max_statutes}, "
        f"min_kept={min_kept}, max_precedents={max_precedents}, "
        f"query_terms_mode={settings.search_query_terms_mode}"
    )
    if memory_enabled:
        print(
            "💾 [Retrieval] Claim-context memory enabled"
            + (" (overwrite requested)" if memory_overwrite else "")
        )

    cache_signature = _build_claim_context_memory_signature() if memory_enabled else {}
    cache_client = get_claim_context_memory() if memory_enabled else None
    if memory_enabled and not memory_overwrite and cache_client is not None:
        _check_cancel()
        progress_callback("Controllo memoria contesto claim", 12)
        cached = cache_client.get(
            claim=claim,
            include_precedents=bool(include_precedents),
            max_statutes=int(max_statutes),
            max_precedents=int(max_precedents),
            signature=cache_signature,
        )
        if isinstance(cached, dict):
            cached_statutes = [
                dict(item)
                for item in (cached.get("statutes") or [])
                if isinstance(item, dict)
            ]
            cached_precedents = [
                dict(item)
                for item in (cached.get("precedents") or [])
                if isinstance(item, dict)
            ]
            print(
                "💾 [Retrieval] Claim-context memory HIT: "
                f"{len(cached_statutes)} statutes, {len(cached_precedents)} precedents"
            )
            _log_claim_context_items(
                claim=claim,
                statutes=cached_statutes,
                precedents=cached_precedents,
                source_label="cache HIT",
            )
            if isinstance(result_metadata, dict):
                result_metadata["memory_enabled"] = True
                result_metadata["memory_hit"] = True
            progress_callback("Memoria contesto claim: cache hit", 66)
            return cached_statutes, cached_precedents

    pipe = get_pipeline()
    retrieval_agent = get_retrieval_filter_agent()
    retrieval_agent.set_cancel_checker(cancel_checker)

    # ── Step 1: classify + embed once ──────────────────────────────────
    _check_cancel()
    progress_callback("Classificazione claim e embedding query", 18)
    classification = pipe.classifier.classify(claim)
    embedding = pipe.embed_text(claim)
    libri_filters = pipe.build_search_filters(
        classification, settings.search_use_top_n_libri
    )

    # ── Step 2: initial hybrid retrieval ───────────────────────────────
    _check_cancel()
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
    article_by_id = {a.statute_id: a for a in articles}
    statutes = _articles_to_dicts(articles)
    _check_cancel()
    legal_context = retrieval_agent._extract_legal_context(claim)
    progress_callback("Filtro rilevanza norme", 38)
    kept_statutes = retrieval_agent.filter_irrelevant_statutes(claim, statutes)
    progress_callback("Verifica applicabilità norme", 48)
    kept_statutes = retrieval_agent.filter_applicable_statutes(
        claim, kept_statutes, legal_context
    )
    seen_ids = {s["statute_id"] for s in statutes}  # all fetched so far

    if kept_statutes:
        _check_cancel()
        seed_articles = [
            article_by_id[s["statute_id"]]
            for s in kept_statutes
            if s.get("statute_id") in article_by_id
        ]
        expanded_articles = pipe.expand_with_cited_articles(seed_articles)
        expanded_only = [
            a
            for a in expanded_articles
            if a.statute_id not in {s.statute_id for s in seed_articles}
        ]
        if expanded_only:
            print(
                "ℹ️ [Retrieval] 🔗 Citation expansion after filters (initial): "
                f"seed={len(seed_articles)}, +{len(expanded_only)} candidates"
            )
            _log_retrieval_debug(
                claim=claim,
                filters=libri_filters,
                articles=expanded_articles,
                stage=f"initial_top_k_{current_top_k}_kept_seed_plus_cites",
            )
            expanded_statutes = [
                d
                for d in _articles_to_dicts(expanded_only)
                if d["statute_id"] not in seen_ids
            ]
            if expanded_statutes:
                _check_cancel()
                seen_ids.update(s["statute_id"] for s in expanded_statutes)
                progress_callback("Filtro rilevanza norme (CITES)", 49)
                kept_from_cites = retrieval_agent.filter_irrelevant_statutes(
                    claim, expanded_statutes
                )
                progress_callback("Verifica applicabilità norme (CITES)", 50)
                kept_from_cites = retrieval_agent.filter_applicable_statutes(
                    claim, kept_from_cites, legal_context
                )
                kept_statutes.extend(kept_from_cites)
                print(
                    "ℹ️ [Retrieval] 📎 CITES filtered result (initial): "
                    f"+{len(kept_from_cites)} kept from {len(expanded_statutes)} new candidates"
                )

    print(
        f"📊 Initial search: {len(statutes)} direct fetched, "
        f"{len(kept_statutes)} kept (min={min_kept})"
    )

    # ── Step 3: progressive expansion ──────────────────────────────────
    expansion = 0
    zero_gain_rounds = 0
    while len(kept_statutes) < min_kept and expansion < max_expansions:
        _check_cancel()
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
        article_by_id = {a.statute_id: a for a in articles}
        new_statutes = [
            d for d in _articles_to_dicts(articles) if d["statute_id"] not in seen_ids
        ]

        if not new_statutes:
            print("   ⚠️ No new articles found — stopping expansion")
            break

        seen_ids.update(s["statute_id"] for s in new_statutes)
        progress_callback("Filtro rilevanza norme (espansione)", 60)
        new_kept = retrieval_agent.filter_irrelevant_statutes(claim, new_statutes)
        progress_callback("Verifica applicabilità norme (espansione)", 62)
        new_kept = retrieval_agent.filter_applicable_statutes(
            claim, new_kept, legal_context
        )

        kept_from_cites = []
        if new_kept:
            _check_cancel()
            seed_articles = [
                article_by_id[s["statute_id"]]
                for s in new_kept
                if s.get("statute_id") in article_by_id
            ]
            expanded_articles = pipe.expand_with_cited_articles(seed_articles)
            seed_ids = {s.statute_id for s in seed_articles}
            expanded_only = [
                a for a in expanded_articles if a.statute_id not in seed_ids
            ]
            if expanded_only:
                print(
                    "ℹ️ [Retrieval] 🔗 Citation expansion after filters "
                    f"(round {expansion}): seed={len(seed_articles)}, +{len(expanded_only)} candidates"
                )
                _log_retrieval_debug(
                    claim=claim,
                    filters=libri_filters,
                    articles=expanded_articles,
                    stage=(
                        f"expansion_{expansion}_top_k_{current_top_k}"
                        "_kept_seed_plus_cites"
                    ),
                )
                expanded_statutes = [
                    d
                    for d in _articles_to_dicts(expanded_only)
                    if d["statute_id"] not in seen_ids
                ]
                if expanded_statutes:
                    _check_cancel()
                    seen_ids.update(s["statute_id"] for s in expanded_statutes)
                    progress_callback("Filtro rilevanza norme (CITES esp.)", 63)
                    kept_from_cites = retrieval_agent.filter_irrelevant_statutes(
                        claim, expanded_statutes
                    )
                    progress_callback("Verifica applicabilità norme (CITES esp.)", 64)
                    kept_from_cites = retrieval_agent.filter_applicable_statutes(
                        claim, kept_from_cites, legal_context
                    )
                    print(
                        "ℹ️ [Retrieval] 📎 CITES filtered result "
                        f"(round {expansion}): +{len(kept_from_cites)} kept from "
                        f"{len(expanded_statutes)} new candidates"
                    )

        round_total_kept = len(new_kept) + len(kept_from_cites)
        kept_statutes.extend(new_kept)
        kept_statutes.extend(kept_from_cites)

        print(
            f"   📊 +{len(new_statutes)} new fetched, "
            f"+{round_total_kept} kept (direct={len(new_kept)}, cites={len(kept_from_cites)}) "
            f"→ total kept={len(kept_statutes)}"
        )

        if round_total_kept == 0:
            zero_gain_rounds += 1
        else:
            zero_gain_rounds = 0

        if (
            zero_gain_rounds >= settings.search_expansion_max_zero_gain_rounds
            and len(kept_statutes) < min_kept
        ):
            print(
                "   ⚠️ Expansion early-stop: nessun guadagno utile in "
                f"{zero_gain_rounds} round consecutivi"
            )
            break

    if expansion > 0:
        print(
            f"✅ Progressive search complete: {expansion} expansion(s), "
            f"{len(kept_statutes)} statutes kept"
        )

    statutes = kept_statutes

    # ── Precedents (unchanged) ─────────────────────────────────────────
    precedents: list[dict] = []
    if include_precedents:
        _check_cancel()
        progress_callback("Recupero precedenti", 64)
        try:
            result = search_precedents_tool.invoke(
                {"query": claim, "limit": max_precedents}
            )
            if isinstance(result, list):
                precedents = result
        except Exception as e:
            print(f"⚠️ Errore recupero precedenti: {e}")

    _check_cancel()
    progress_callback("Filtro precedenti", 66)
    precedents = retrieval_agent.filter_irrelevant_precedents(claim, precedents)

    if memory_enabled and cache_client is not None:
        try:
            cache_key = cache_client.put(
                claim=claim,
                include_precedents=bool(include_precedents),
                max_statutes=int(max_statutes),
                max_precedents=int(max_precedents),
                signature=cache_signature,
                statutes=statutes,
                precedents=precedents,
            )
            print(
                "💾 [Retrieval] Claim-context memory "
                f"{'REFRESHED' if memory_overwrite else 'SAVED'}: "
                f"{len(statutes)} statutes, {len(precedents)} precedents "
                f"(key={cache_key[:10]}...)"
            )
            if isinstance(result_metadata, dict):
                result_metadata["memory_saved"] = True
                result_metadata["cache_key"] = cache_key
        except Exception as e:
            print(f"⚠️ [Retrieval] Claim-context memory save failed: {e}")

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
        router_agent = Router(
            config=AgentConfig(
                temperature=PIPELINE_AUX_LLM_TEMPERATURE,
                max_tokens=settings.llm_max_tokens,
            )
        )
        print("✅ Router pronto!")
    return router_agent


def get_polisher_evaluator():
    """Lazy load del Polisher-Evaluator agent."""
    global polisher_evaluator
    if polisher_evaluator is None:
        print("🔧 Inizializzazione Polisher-Evaluator...")
        polisher_evaluator = PolisherEvaluator(
            config=AgentConfig(
                temperature=PIPELINE_AUX_LLM_TEMPERATURE,
                max_tokens=settings.llm_max_tokens,
            )
        )
        print("✅ Polisher-Evaluator pronto!")
    return polisher_evaluator


def resolve_routing_decision(
    claim: str, payload: dict | None = None, cancel_checker=None
) -> RoutingDecision:
    """
    Determina il routing (causal_type_id/theory_id) usando eventuali hint del payload,
    altrimenti invoca il Router.
    """
    router = get_router()
    cancel_checker = cancel_checker or (lambda: False)
    router.set_cancel_checker(cancel_checker)
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
        domain = str(routing_hint.get("domain", "")).strip().upper()
        if domain not in ("CIVILE", "PENALE", "AMMINISTRATIVO", "ENTRAMBI"):
            if cancel_checker():
                raise PipelineCancelled("Esecuzione interrotta manualmente.")
            try:
                domain = router.route(claim).domain
            except Exception:
                domain = "ENTRAMBI"
        return RoutingDecision(
            claim=claim,
            domain=domain,
            causal_type_id=ct_valid,
            theory_id=th_valid or "",
            anchor_norms=config_loader.anchor_norms_for(ct_valid),
            principle_tests=config_loader.principle_tests_for(ct_valid),
            additional_causal_types=[],
        )

    if cancel_checker():
        raise PipelineCancelled("Esecuzione interrotta manualmente.")
    return router.route(claim)


def _clone_context_items(items: list[dict]) -> list[dict]:
    """Shallow-copy retrieved context items so agents can annotate independently."""
    return [dict(item) for item in (items or [])]


def _coerce_bool(value, default: bool = False) -> bool:
    """Parse bool-like values from API payloads with safe defaults."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _parse_claim_context_memory_flags(data: dict | None) -> tuple[bool, bool]:
    payload = data if isinstance(data, dict) else {}
    enabled = bool(payload.get("claim_context_memory_enabled", False))
    overwrite = bool(payload.get("claim_context_memory_overwrite", False))
    if overwrite and not enabled:
        enabled = True
    return enabled, overwrite


def _parse_retrieval_model_order_override(value) -> list[str] | None:
    """Parse request-scoped retrieval model alias order override."""
    if value is None:
        return None
    aliases: list[str] = []
    if isinstance(value, str):
        aliases = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        aliases = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(
            "settings.retrieval_model_order_aliases must be a list or string"
        )

    if not aliases:
        return None

    available = set(settings.available_model_aliases)
    invalid = [alias for alias in aliases if alias not in available]
    if invalid:
        raise ValueError(
            "Invalid retrieval model alias(es): "
            + ", ".join(invalid)
            + ". Available: "
            + ", ".join(sorted(available))
        )
    return aliases


@contextmanager
def _temporary_retrieval_model_order_override(aliases: list[str] | None):
    """Temporarily override retrieval model fallback alias order for a request."""
    if not aliases:
        yield
        return

    with _retrieval_model_override_lock:
        cfg_cls = settings.__class__
        previous = list(cfg_cls.RETRIEVAL_MODEL_ORDER_ALIASES)
        cfg_cls.RETRIEVAL_MODEL_ORDER_ALIASES = list(aliases)
        try:
            print(
                "⚙️  [Retrieval] Model order override (request): "
                + " -> ".join(cfg_cls.RETRIEVAL_MODEL_ORDER_ALIASES)
            )
            yield
        finally:
            cfg_cls.RETRIEVAL_MODEL_ORDER_ALIASES = previous


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
                "reasoner_temperature": settings.reasoner_default_temperature,
                "counter_temperature": settings.counter_default_temperature,
                "llm_max_tokens": settings.llm_max_tokens,
                "search_top_k_default": settings.search_top_k_default,
                "search_min_kept_statutes": settings.search_min_kept_statutes,
                "search_use_top_n_libri": settings.search_use_top_n_libri,
                "search_query_terms_mode": settings.search_query_terms_mode,
                "search_query_terms_llm_max_terms": settings.search_query_terms_llm_max_terms,
                "search_query_terms_llm_max_tokens": settings.search_query_terms_llm_max_tokens,
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
                "reasoner_enable_causality": True,
                "counter_enable_causality": False,
                "counter_pass_causal_identity": False,
                "counter_pass_taxonomy_attacks": False,
                "counter_pass_norms": False,
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
        include_precedents = data.get("include_precedents", True)
        max_precedents = data.get("max_precedents", settings.precedents_limit_default)
        claim_memory_enabled, claim_memory_overwrite = (
            _parse_claim_context_memory_flags(data)
        )
        fe_settings = data.get("settings", {}) or {}
        fe_search_query_terms_mode = fe_settings.get("search_query_terms_mode")
        fe_search_min_kept_statutes = fe_settings.get("search_min_kept_statutes")
        fe_search_use_top_n_libri = fe_settings.get("search_use_top_n_libri")
        fe_retrieval_model_order_aliases = fe_settings.get(
            "retrieval_model_order_aliases"
        )
        fe_retrieval_strict_llm_errors = fe_settings.get("retrieval_strict_llm_errors")

        if not claim:
            return jsonify({"error": 'Campo "message" obbligatorio'}), 400

        if fe_search_query_terms_mode is not None:
            mode = str(fe_search_query_terms_mode).strip().lower()
            if mode == "llm":
                settings.search_query_terms_mode = mode
        if fe_search_min_kept_statutes is not None:
            settings.search_min_kept_statutes = int(fe_search_min_kept_statutes)
        if fe_search_use_top_n_libri is not None:
            settings.search_use_top_n_libri = int(fe_search_use_top_n_libri)

        retrieval_model_order_override = _parse_retrieval_model_order_override(
            fe_retrieval_model_order_aliases
        )
        retrieval_strict_llm_errors = bool(fe_retrieval_strict_llm_errors)

        with (
            _temporary_retrieval_model_order_override(retrieval_model_order_override),
            retrieval_llm_fail_fast_scope(retrieval_strict_llm_errors),
        ):
            # Align /api/chat retrieval with the full pipeline retrieval stack:
            # hybrid retrieval + CITES expansion + relevance filter + applicability filter
            statutes, precedents = prepare_claim_context(
                claim=claim,
                include_precedents=bool(include_precedents),
                max_statutes=int(top_k),
                max_precedents=int(max_precedents),
                claim_context_memory_enabled=claim_memory_enabled,
                claim_context_memory_overwrite=claim_memory_overwrite,
            )
            classification = pipe.classifier.classify(claim)
            chat_result = SimpleNamespace(
                claim=claim,
                classification=classification,
                articles=[
                    SimpleNamespace(
                        source=art.get("source"),
                        articolo=art.get("articolo"),
                        titolo=art.get("titolo"),
                        testo=art.get("testo"),
                        libro=art.get("libro"),
                        score=float(art.get("score", 0.0)),
                    )
                    for art in statutes
                ],
            )
            response_text = format_search_result(chat_result)

        return jsonify(
            {
                "response": response_text,
                "classification": {
                    "categories": classification.categories,
                    "descriptions": classification.descriptions,
                    "libro_mappings": classification.libro_mappings,
                },
                "articles": [
                    {
                        "source": art.get("source"),
                        "articolo": art.get("articolo"),
                        "titolo": art.get("titolo"),
                        "testo": art.get("testo"),
                        "libro": art.get("libro"),
                        "score": float(art.get("score", 0.0)),
                    }
                    for art in statutes
                ],
                "precedents": precedents,
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
        fe_settings = data.get("settings", {}) or {}
        fe_reasoner_temperature = fe_settings.get(
            "reasoner_temperature",
            fe_settings.get("llm_temperature", settings.reasoner_default_temperature),
        )
        fe_legacy_enable_causality = fe_settings.get("enable_causality")
        fe_reasoner_enable_causality = _coerce_bool(
            fe_settings.get(
                "reasoner_enable_causality",
                (
                    True
                    if fe_legacy_enable_causality is None
                    else _coerce_bool(fe_legacy_enable_causality, True)
                ),
            ),
            True,
        )
        fe_max_tokens = fe_settings.get("llm_max_tokens")
        fe_reasoner_model = fe_settings.get("reasoner_model")
        include_precedents = data.get("include_precedents", True)
        max_statutes = data.get("max_statutes", settings.search_top_k_default)
        max_precedents = data.get("max_precedents", settings.precedents_limit_default)
        claim_memory_enabled, claim_memory_overwrite = (
            _parse_claim_context_memory_flags(data)
        )
        if not claim:
            return jsonify({"error": 'Campo "claim" obbligatorio'}), 400

        routing_decision = resolve_routing_decision(claim, data)
        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            claim_context_memory_enabled=claim_memory_enabled,
            claim_context_memory_overwrite=claim_memory_overwrite,
        )
        reasoner_config = _build_agent_config(
            model_override=fe_reasoner_model or settings.reasoner_default_model,
            temperature=fe_reasoner_temperature,
            max_tokens=fe_max_tokens,
        )
        reas = Reasoner(config=reasoner_config)

        result = reas.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=statutes,
            pre_retrieved_precedents=precedents,
            enable_causality=fe_reasoner_enable_causality,
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


def _run_counter_reason_only(
    data: dict,
    *,
    status_callback=None,
    token_callback=None,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Execute Counter-Reasoner only (JSON + SSE shared implementation)."""
    if not isinstance(data, dict):
        data = {}

    status_callback = status_callback or (lambda _msg: None)
    token_callback = token_callback or (lambda _payload: None)
    progress_callback = progress_callback or (lambda _event, _payload: None)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")

    def _emit_progress(event_name: str, payload: dict) -> None:
        _check_cancel()
        try:
            progress_callback(event_name, payload)
        except PipelineCancelled:
            raise
        except Exception:
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
    fe_settings = data.get("settings", {}) or {}
    fe_counter_temperature = fe_settings.get(
        "counter_temperature",
        fe_settings.get("llm_temperature", settings.counter_default_temperature),
    )
    fe_legacy_enable_causality = fe_settings.get("enable_causality")
    fe_legacy_mode = (
        "enable_causality" in fe_settings
        and "reasoner_enable_causality" not in fe_settings
        and "counter_enable_causality" not in fe_settings
        and "counter_pass_causal_identity" not in fe_settings
        and "counter_pass_taxonomy_attacks" not in fe_settings
        and "counter_pass_norms" not in fe_settings
    )
    fe_counter_enable_causality = _coerce_bool(
        fe_settings.get(
            "counter_enable_causality",
            (
                True
                if fe_legacy_enable_causality is None
                else _coerce_bool(fe_legacy_enable_causality, True)
            ),
        ),
        True,
    )
    default_counter_pass = (
        _coerce_bool(fe_legacy_enable_causality, False) if fe_legacy_mode else False
    )
    fe_counter_pass_causal_identity = _coerce_bool(
        fe_settings.get("counter_pass_causal_identity", default_counter_pass),
        default_counter_pass,
    )
    fe_counter_pass_taxonomy_attacks = _coerce_bool(
        fe_settings.get("counter_pass_taxonomy_attacks", default_counter_pass),
        default_counter_pass,
    )
    fe_counter_pass_norms = _coerce_bool(
        fe_settings.get("counter_pass_norms", default_counter_pass),
        default_counter_pass,
    )
    fe_max_tokens = fe_settings.get("llm_max_tokens")
    fe_counter_model = fe_settings.get("counter_model")
    include_precedents = data.get("include_precedents", True)
    max_statutes = data.get("max_statutes", settings.search_top_k_default)
    max_precedents = data.get("max_precedents", settings.precedents_limit_default)
    reasoner_conclusion = (data.get("reasoner_conclusion", "") or "").strip()
    claim_memory_enabled, claim_memory_overwrite = _parse_claim_context_memory_flags(
        data
    )

    if not claim:
        raise ValueError('Campo "claim" obbligatorio')
    if not reasoner_conclusion:
        raise ValueError(
            'Campo "reasoner_conclusion" obbligatorio per il Counter-Reasoner'
        )

    _check_cancel()
    status_callback("Preparazione contesto counter...")
    _emit_phase("context_setup", "active", 12, "Routing e setup Counter-Reasoner")

    routing_decision = resolve_routing_decision(claim, data)
    _emit_phase("context_setup", "active", 28, "Recupero contesto condiviso")

    retrieval_context_meta: dict = {}
    pre_retrieved_statutes = data.get("pre_retrieved_statutes")
    pre_retrieved_precedents = data.get("pre_retrieved_precedents")
    if isinstance(pre_retrieved_statutes, list) and isinstance(
        pre_retrieved_precedents, list
    ):
        statutes = _clone_context_items(pre_retrieved_statutes)
        precedents = _clone_context_items(pre_retrieved_precedents)
        print(
            "ℹ️ [CounterReason] Using pre-retrieved shared context: "
            f"{len(statutes)} statutes, {len(precedents)} precedents"
        )
    else:
        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            claim_context_memory_enabled=claim_memory_enabled,
            claim_context_memory_overwrite=claim_memory_overwrite,
            progress_callback=lambda detail, progress: _emit_phase(
                "context_setup", "active", progress, detail
            ),
            result_metadata=retrieval_context_meta,
            cancel_checker=lambda: cancel_event is not None and cancel_event.is_set(),
        )

    _emit_progress(
        "retrieval_context",
        {
            "statutes": _clone_context_items(statutes),
            "precedents": _clone_context_items(precedents),
            "memory": {
                "enabled": bool(retrieval_context_meta.get("memory_enabled"))
                or bool(claim_memory_enabled),
                "overwrite": bool(retrieval_context_meta.get("memory_overwrite"))
                or bool(claim_memory_overwrite),
                "hit": bool(retrieval_context_meta.get("memory_hit")),
            },
        },
    )
    _emit_phase("context_setup", "done", 100, "Contesto counter pronto")
    _emit_phase("counter", "active", 8, "Generazione contro-argomentazione")
    status_callback("Generazione contro-argomentazione in corso...")

    counter_routing_decision = RoutingDecision(
        claim=claim,
        domain=routing_decision.domain,
        causal_type_id=(
            routing_decision.causal_type_id if fe_counter_pass_causal_identity else ""
        ),
        theory_id=(
            routing_decision.theory_id if fe_counter_pass_causal_identity else ""
        ),
        anchor_norms=(
            routing_decision.anchor_norms
            if (fe_counter_enable_causality and fe_counter_pass_norms)
            else {}
        ),
        principle_tests=(
            routing_decision.principle_tests
            if (fe_counter_enable_causality and fe_counter_pass_norms)
            else []
        ),
        additional_causal_types=(
            routing_decision.additional_causal_types
            if fe_counter_pass_causal_identity
            else []
        ),
    )
    counter_enable_causality_effective = (
        fe_counter_enable_causality
        and fe_counter_pass_taxonomy_attacks
        and fe_counter_pass_causal_identity
    )

    counter_config = _build_agent_config(
        model_override=fe_counter_model or settings.counter_default_model,
        temperature=fe_counter_temperature,
        max_tokens=fe_max_tokens,
    )
    cr = CounterReasoner(config=counter_config)
    cr.set_cancel_checker(lambda: cancel_event is not None and cancel_event.is_set())

    def _counter_stream_callback(payload: dict) -> None:
        _check_cancel()
        token_callback(payload)

    result = cr.run(
        claim=claim,
        routing_decision=counter_routing_decision,
        pre_retrieved_statutes=_clone_context_items(statutes),
        pre_retrieved_precedents=_clone_context_items(precedents),
        enable_causality=counter_enable_causality_effective,
        reasoner_conclusion=reasoner_conclusion,
        stream_callback=_counter_stream_callback,
    )

    _emit_phase("counter", "active", 97, "Costruzione ASPIC+ e output finale")
    payload = result.to_dict()
    _emit_progress("counter_result", payload)
    _emit_phase("counter", "done", 100, "Contro-argomentazione completata")
    status_callback("Counter-Reasoner completato.")
    return payload


@app.route("/api/counter_reason", methods=["POST"])
def counter_reason():
    """
    Endpoint per il contro-ragionamento.
    """
    try:
        data = request.get_json(silent=True) or {}
        result = _run_counter_reason_only(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Errore counter-reasoning: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/counter_reason/stream", methods=["POST"])
def counter_reason_stream():
    """Endpoint SSE con streaming token-by-token del solo Counter-Reasoner."""
    if not _pipeline_lock.acquire(blocking=False):
        return jsonify({"error": "A pipeline is already running. Please wait."}), 429

    data = request.get_json(silent=True) or {}
    event_queue: Queue = Queue()
    sentinel = object()
    run_id = uuid.uuid4().hex
    cancel_event = threading.Event()

    global _active_stream_run
    with _active_stream_run_lock:
        _active_stream_run = {
            "run_id": run_id,
            "cancel_event": cancel_event,
            "started_at": time.time(),
            "kind": "counter_reason_stream",
        }

    def push_status(message: str) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put(("status", {"message": message}))

    def push_token(payload: dict) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put(("token", payload))

    def push_progress(event_name: str, payload: dict) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put((event_name, payload))

    def push_error(message: str, code: int) -> None:
        event_queue.put(("error", {"message": message, "code": code}))

    def _worker() -> None:
        global _active_stream_run
        try:
            result = _run_counter_reason_only(
                data,
                status_callback=push_status,
                token_callback=push_token,
                progress_callback=push_progress,
                cancel_event=cancel_event,
            )
            event_queue.put(("final", result))
        except PipelineCancelled as e:
            event_queue.put(
                ("cancelled", {"message": str(e), "run_id": run_id, "ok": True})
            )
        except ValueError as e:
            push_error(str(e), 400)
        except Exception as e:
            print(f"\n{'='*70}")
            print("❌ COUNTER STREAM ERROR")
            print(f"{'='*70}")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            push_error(str(e), 500)
        finally:
            event_queue.put(("done", {"ok": True}))
            event_queue.put(sentinel)
            with _active_stream_run_lock:
                if (
                    isinstance(_active_stream_run, dict)
                    and _active_stream_run.get("run_id") == run_id
                ):
                    _active_stream_run = None
            _pipeline_lock.release()

    threading.Thread(target=_worker, daemon=True).start()
    event_queue.put(("run_started", {"run_id": run_id}))

    @stream_with_context
    def generate():
        while True:
            try:
                item = event_queue.get(timeout=12)
            except Empty:
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


def _sse_event(event: str, payload: dict) -> str:
    """Encode one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_full_pipeline(
    data: dict,
    *,
    status_callback=None,
    token_callback=None,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Execute the full pipeline and optionally emit status/token callbacks."""
    if not isinstance(data, dict):
        data = {}

    status_callback = status_callback or (lambda _msg: None)
    token_callback = token_callback or (lambda _payload: None)
    progress_callback = progress_callback or (lambda _event, _payload: None)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")

    def _is_cancel_requested() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _emit_progress(event_name: str, payload: dict) -> None:
        _check_cancel()
        try:
            progress_callback(event_name, payload)
        except PipelineCancelled:
            raise
        except Exception:
            # Progress streaming must never break pipeline execution.
            pass

    def _emit_phase(
        phase: str, status: str, progress: int, detail: str | None = None
    ) -> None:
        _check_cancel()
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
    claim_memory_enabled, claim_memory_overwrite = _parse_claim_context_memory_flags(
        data
    )

    # ── Frontend-configurable settings ────────────────────────────────
    fe_settings = data.get("settings", {}) or {}
    fe_reasoner_temperature = fe_settings.get(
        "reasoner_temperature",
        fe_settings.get("llm_temperature", settings.reasoner_default_temperature),
    )
    fe_counter_temperature = fe_settings.get(
        "counter_temperature",
        fe_settings.get("llm_temperature", settings.counter_default_temperature),
    )
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
    fe_search_use_top_n_libri = fe_settings.get("search_use_top_n_libri")
    fe_chain_min_steps = fe_settings.get("chain_min_steps")
    fe_chain_max_steps = fe_settings.get("chain_max_steps")
    fe_search_query_terms_mode = fe_settings.get("search_query_terms_mode")
    fe_search_query_terms_llm_max_terms = fe_settings.get(
        "search_query_terms_llm_max_terms"
    )
    fe_search_query_terms_llm_max_tokens = fe_settings.get(
        "search_query_terms_llm_max_tokens"
    )
    fe_legacy_enable_causality = fe_settings.get("enable_causality")
    fe_legacy_mode = (
        "enable_causality" in fe_settings
        and "reasoner_enable_causality" not in fe_settings
        and "counter_enable_causality" not in fe_settings
        and "counter_pass_causal_identity" not in fe_settings
        and "counter_pass_taxonomy_attacks" not in fe_settings
        and "counter_pass_norms" not in fe_settings
    )
    fe_reasoner_enable_causality = _coerce_bool(
        fe_settings.get(
            "reasoner_enable_causality",
            (
                True
                if fe_legacy_enable_causality is None
                else _coerce_bool(fe_legacy_enable_causality, True)
            ),
        ),
        True,
    )
    fe_counter_enable_causality = _coerce_bool(
        fe_settings.get(
            "counter_enable_causality",
            (
                True
                if fe_legacy_enable_causality is None
                else _coerce_bool(fe_legacy_enable_causality, True)
            ),
        ),
        True,
    )
    default_counter_pass = (
        _coerce_bool(fe_legacy_enable_causality, False) if fe_legacy_mode else False
    )
    fe_counter_pass_causal_identity = _coerce_bool(
        fe_settings.get("counter_pass_causal_identity", default_counter_pass),
        default_counter_pass,
    )
    fe_counter_pass_taxonomy_attacks = _coerce_bool(
        fe_settings.get("counter_pass_taxonomy_attacks", default_counter_pass),
        default_counter_pass,
    )
    fe_counter_pass_norms = _coerce_bool(
        fe_settings.get("counter_pass_norms", default_counter_pass),
        default_counter_pass,
    )

    if not claim:
        raise ValueError('Campo "claim" obbligatorio')

    _check_cancel()
    status_callback("Avvio pipeline completa...")
    _emit_phase("context_setup", "active", 5, "Routing iniziale")

    with _pipeline_logger(claim):
        print(f"\n{'='*70}")
        print("🚀 FULL PIPELINE - START")
        print(f"{'='*70}")
        print(f"Claim: {claim[:100]}...")

        if fe_settings:
            print(f"⚙️  Frontend settings override: {fe_settings}")
        if not fe_reasoner_enable_causality:
            print("🔬 Reasoner causality taxonomy DISABLED by frontend settings")
        if not fe_counter_enable_causality:
            print("🔬 Counter causality taxonomy DISABLED by frontend settings")
        if not fe_counter_pass_causal_identity:
            print("🧩 Counter input: causal_type_id/theory_id pass-through DISABLED")
        if not fe_counter_pass_taxonomy_attacks:
            print("⚔️ Counter input: taxonomy attack pool pass-through DISABLED")
        if not fe_counter_pass_norms:
            print(
                "📚 Counter input: taxonomy anchor norms/principle tests pass-through DISABLED"
            )
        if claim_memory_enabled:
            print(
                "💾 Claim-context memory ENABLED"
                + (" (overwrite requested)" if claim_memory_overwrite else "")
            )

        _check_cancel()
        # Apply chain step overrides to global settings
        if fe_chain_min_steps is not None:
            settings.chain_min_steps = int(fe_chain_min_steps)
        if fe_chain_max_steps is not None:
            settings.chain_max_steps = int(fe_chain_max_steps)
        if fe_search_min_kept_statutes is not None:
            settings.search_min_kept_statutes = int(fe_search_min_kept_statutes)
        if fe_search_use_top_n_libri is not None:
            settings.search_use_top_n_libri = int(fe_search_use_top_n_libri)
        if fe_search_query_terms_mode is not None:
            mode = str(fe_search_query_terms_mode).strip().lower()
            if mode == "llm":
                settings.search_query_terms_mode = mode
        if fe_search_query_terms_llm_max_terms is not None:
            settings.search_query_terms_llm_max_terms = int(
                fe_search_query_terms_llm_max_terms
            )
        if fe_search_query_terms_llm_max_tokens is not None:
            settings.search_query_terms_llm_max_tokens = int(
                fe_search_query_terms_llm_max_tokens
            )

        _check_cancel()
        status_callback("Preparazione contesto giuridico...")
        _emit_phase("context_setup", "active", 15, "Routing dominio e causalità")
        routing_decision = resolve_routing_decision(
            claim,
            data,
            cancel_checker=_is_cancel_requested,
        )
        _emit_phase("context_setup", "active", 20, "Avvio recupero contesto")

        # Preload context once for both reasoners
        def _emit_context_detail(detail: str, progress: int) -> None:
            _emit_phase("context_setup", "active", progress, detail)

        retrieval_context_meta: dict = {}
        statutes, precedents = prepare_claim_context(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            claim_context_memory_enabled=claim_memory_enabled,
            claim_context_memory_overwrite=claim_memory_overwrite,
            progress_callback=_emit_context_detail,
            cancel_checker=_is_cancel_requested,
            result_metadata=retrieval_context_meta,
        )
        _check_cancel()
        _emit_progress(
            "retrieval_context",
            {
                "statutes": _clone_context_items(statutes),
                "precedents": _clone_context_items(precedents),
                "memory": {
                    "enabled": bool(retrieval_context_meta.get("memory_enabled")),
                    "overwrite": bool(retrieval_context_meta.get("memory_overwrite")),
                    "hit": bool(retrieval_context_meta.get("memory_hit")),
                },
            },
        )
        _emit_phase("context_setup", "active", 68, "Preparazione contesto agenti")
        reasoner_statutes = _clone_context_items(statutes)
        reasoner_precedents = _clone_context_items(precedents)
        # Counter always receives pre-retrieval KB (statutes + precedents).
        # Taxonomy-specific norms are controlled separately via counter_pass_norms.
        counter_statutes = _clone_context_items(statutes)
        counter_precedents = _clone_context_items(precedents)

        # STEP 1: primary reasoning on the claim
        print(f"\n{'─'*70}")
        print("📊 STEP 1: Reasoner execution (shared retrieved context)...")
        print(f"{'─'*70}")
        print(
            f"   📚 Knowledge base: {len(reasoner_statutes)} statutes, {len(reasoner_precedents)} precedents"
        )
        _emit_phase("context_setup", "done", 100, "Contesto pronto")
        _emit_phase("support", "active", 8, "Generazione ragionamento")
        status_callback("Generazione argomentazione principale in corso...")

        _check_cancel()
        reasoner_config = _build_agent_config(
            model_override=fe_reasoner_model or settings.reasoner_default_model,
            temperature=fe_reasoner_temperature,
            max_tokens=fe_max_tokens,
        )
        reas = Reasoner(config=reasoner_config)
        reas.set_cancel_checker(_is_cancel_requested)

        def _reasoner_stream_callback(payload: dict) -> None:
            """Forward reasoner tokens and translate control frames into SSE events."""
            if isinstance(payload, dict):
                control_event = str(payload.get("_control_event", "") or "").strip()
                if control_event in {
                    "reasoner_refinement_started",
                    "reasoner_refinement_completed",
                }:
                    _emit_progress(control_event, payload.get("payload") or {})
                    return
            token_callback(payload)

        reasoner_result = reas.run(
            claim=claim,
            routing_decision=routing_decision,
            pre_retrieved_statutes=reasoner_statutes,
            pre_retrieved_precedents=reasoner_precedents,
            enable_causality=fe_reasoner_enable_causality,
            stream_callback=_reasoner_stream_callback,
        )
        _check_cancel()
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
        counter_routing_decision = RoutingDecision(
            claim=claim,
            domain=routing_decision.domain,
            causal_type_id=(
                reasoner_result.causal_type_id
                if fe_counter_pass_causal_identity
                else ""
            ),
            theory_id=(
                reasoner_result.theory_id if fe_counter_pass_causal_identity else ""
            ),
            anchor_norms=(
                reasoner_result.anchor_norms
                if (fe_counter_enable_causality and fe_counter_pass_norms)
                else {}
            ),
            principle_tests=(
                reasoner_result.principle_tests
                if (fe_counter_enable_causality and fe_counter_pass_norms)
                else []
            ),
            additional_causal_types=(
                reasoner_result.causal_type_ids_for_counter
                if fe_counter_pass_causal_identity
                else []
            ),
        )
        counter_enable_causality_effective = (
            fe_counter_enable_causality
            and fe_counter_pass_taxonomy_attacks
            and fe_counter_pass_causal_identity
        )
        if (
            fe_counter_enable_causality
            and fe_counter_pass_taxonomy_attacks
            and not fe_counter_pass_causal_identity
        ):
            print(
                "⚠️ Counter taxonomy requested without causal identity; "
                "falling back to open attacks (enable_causality=False)"
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
        print("⚔️  STEP 2: Counter-Reasoner execution (shared retrieved context)...")
        print(f"{'─'*70}")
        print(
            f"   📚 Knowledge base: {len(counter_statutes)} statutes, {len(counter_precedents)} precedents"
        )
        status_callback("Generazione argomentazione contraria in corso...")

        _check_cancel()
        counter_config = _build_agent_config(
            model_override=fe_counter_model or settings.counter_default_model,
            temperature=fe_counter_temperature,
            max_tokens=fe_max_tokens,
        )
        cr = CounterReasoner(config=counter_config)
        cr.set_cancel_checker(_is_cancel_requested)

        # Opposition consistency is validated by the Polisher gate.
        reasoner_conclusion = reasoner_result.conclusion or ""
        if not reasoner_conclusion:
            raise RuntimeError(
                "Reasoner conclusion is required before Counter-Reasoner execution"
            )
        print(
            "ℹ️ Reasoner conclusion captured for Counter traceability: "
            f"{reasoner_conclusion[:120]}..."
        )

        counter_result = cr.run(
            claim=claim,
            routing_decision=counter_routing_decision,
            pre_retrieved_statutes=counter_statutes,
            pre_retrieved_precedents=counter_precedents,
            enable_causality=counter_enable_causality_effective,
            reasoner_conclusion=reasoner_conclusion,
            stream_callback=token_callback,
        )
        _check_cancel()
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

        _check_cancel()
        pe = get_polisher_evaluator()
        pe.set_cancel_checker(_is_cancel_requested)

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

        _check_cancel()
        evaluation_result = pe.run(
            claim=claim,
            domain=final_routing_decision.domain,
            reasoner_output=reasoner_result.to_dict(),
            counter_reasoner_output=counter_result.to_dict(),
            progress_callback=_emit_progress,
        )

        _check_cancel()

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

        # Derive winning_side and confidence from AQA verdict.
        # Keep backward-compatible labels in `winning_side` while also exposing
        # canonical thesis labels for new consumers.
        aqa = evaluation_result.aqa_report or {}
        aqa_verdict = aqa.get("verdict", "uncertain")
        aqa_net = aqa.get("net_plausibility", {})
        legacy_verdict_map = {
            "plausible": "support",
            "implausible": "counter",
            "uncertain": "undecided",
        }
        canonical_verdict_map = {
            "plausible": "primary_thesis",
            "implausible": "counter_thesis",
            "uncertain": "undecided",
        }
        winning_side_legacy = legacy_verdict_map.get(aqa_verdict, "undecided")
        winning_side_canonical = canonical_verdict_map.get(aqa_verdict, "undecided")

        evaluation_result.winning_side = winning_side_legacy
        evaluation_result.confidence = abs(aqa_net.get("final", 0.0))
        evaluation_payload = evaluation_result.to_dict()
        evaluation_payload["winning_side_canonical"] = winning_side_canonical
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
        print(
            f"   - Winning side: {evaluation_result.winning_side} "
            f"(canonical: {winning_side_canonical})"
        )
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

    _check_cancel()
    status_callback("Pipeline completata.")
    _emit_phase("final_evaluation", "done", 100, "Valutazione completata")
    return {
        "claim": claim,
        "retrieval_context": {
            "statutes": _clone_context_items(statutes),
            "precedents": _clone_context_items(precedents),
            "memory": {
                "enabled": bool(retrieval_context_meta.get("memory_enabled")),
                "overwrite": bool(retrieval_context_meta.get("memory_overwrite")),
                "hit": bool(retrieval_context_meta.get("memory_hit")),
            },
        },
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
    run_id = uuid.uuid4().hex
    cancel_event = threading.Event()

    global _active_stream_run
    with _active_stream_run_lock:
        _active_stream_run = {
            "run_id": run_id,
            "cancel_event": cancel_event,
            "started_at": time.time(),
        }

    def push_status(message: str) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put(("status", {"message": message}))

    def push_token(payload: dict) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put(("token", payload))

    def push_progress(event_name: str, payload: dict) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put((event_name, payload))

    def push_error(message: str, code: int) -> None:
        event_queue.put(("error", {"message": message, "code": code}))

    def _worker() -> None:
        global _active_stream_run
        try:
            result = _run_full_pipeline(
                data,
                status_callback=push_status,
                token_callback=push_token,
                progress_callback=push_progress,
                cancel_event=cancel_event,
            )
            event_queue.put(("final", result))
        except PipelineCancelled as e:
            event_queue.put(
                ("cancelled", {"message": str(e), "run_id": run_id, "ok": True})
            )
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
            with _active_stream_run_lock:
                if (
                    isinstance(_active_stream_run, dict)
                    and _active_stream_run.get("run_id") == run_id
                ):
                    _active_stream_run = None
            _pipeline_lock.release()

    threading.Thread(target=_worker, daemon=True).start()
    event_queue.put(("run_started", {"run_id": run_id}))

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


@app.route("/api/pipeline/stop", methods=["POST"])
def pipeline_stop():
    """
    Richiede l'interruzione della pipeline SSE in esecuzione.
    """
    data = request.get_json(silent=True) or {}
    requested_run_id = str(data.get("run_id", "") or "").strip()

    global _active_stream_run
    with _active_stream_run_lock:
        active = _active_stream_run if isinstance(_active_stream_run, dict) else None
        if not active:
            return jsonify({"ok": False, "error": "No active pipeline run."}), 404

        active_run_id = str(active.get("run_id", "") or "")
        if requested_run_id and active_run_id and requested_run_id != active_run_id:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Run id mismatch with active pipeline.",
                        "active_run_id": active_run_id,
                    }
                ),
                409,
            )

        cancel_event = active.get("cancel_event")
        if isinstance(cancel_event, threading.Event):
            cancel_event.set()

    return jsonify({"ok": True, "run_id": active_run_id})


def _run_evaluate_only(
    data: dict,
    *,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Execute evaluator only (JSON + SSE shared implementation)."""
    if not isinstance(data, dict):
        data = {}

    progress_callback = progress_callback or (lambda _event, _payload: None)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")

    def _emit_progress(event_name: str, payload: dict) -> None:
        _check_cancel()
        try:
            progress_callback(event_name, payload)
        except PipelineCancelled:
            raise
        except Exception:
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
    domain = data.get("domain", "ENTRAMBI")
    reasoner_output = data.get("reasoner_output", {}) or {}
    counter_output = data.get("counter_output", {}) or {}
    fe_settings = data.get("settings", {}) or {}

    if not claim:
        raise ValueError('Campo "claim" obbligatorio')

    _check_cancel()
    _emit_phase("final_evaluation", "active", 10, "Avvio verifica consistenza")

    fe_aqa_alpha = fe_settings.get("aqa_alpha")
    fe_aqa_beta = fe_settings.get("aqa_beta")
    fe_aqa_gamma = fe_settings.get("aqa_gamma")
    fe_aqa_min_semantic_overlap = fe_settings.get("aqa_min_semantic_overlap")
    fe_aqa_min_strength_ratio = fe_settings.get("aqa_min_strength_ratio")
    fe_aqa_damage_factor = fe_settings.get("aqa_damage_factor")
    fe_aqa_allow_factual_attacks = fe_settings.get("aqa_allow_factual_attacks")
    fe_aqa_allow_cross_codice = fe_settings.get("aqa_allow_cross_codice")
    fe_aqa_strength_ratio_by_type = fe_settings.get("aqa_strength_ratio_by_type")

    if fe_aqa_alpha is not None:
        settings.aqa_alpha = float(fe_aqa_alpha)
    if fe_aqa_beta is not None:
        settings.aqa_beta = float(fe_aqa_beta)
    if fe_aqa_gamma is not None:
        settings.aqa_gamma = float(fe_aqa_gamma)
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

    pe = get_polisher_evaluator()
    pe.set_cancel_checker(lambda: cancel_event is not None and cancel_event.is_set())
    result = pe.run(
        claim=claim,
        domain=domain,
        reasoner_output=reasoner_output,
        counter_reasoner_output=counter_output,
        progress_callback=_emit_progress,
    )

    _check_cancel()
    payload = result.to_dict()
    # Keep /api/evaluate aligned with /api/pipeline winning-side semantics.
    aqa = payload.get("aqa_report") or {}
    aqa_verdict = aqa.get("verdict", "uncertain")
    aqa_net = aqa.get("net_plausibility", {}) or {}
    legacy_verdict_map = {
        "plausible": "support",
        "implausible": "counter",
        "uncertain": "undecided",
    }
    canonical_verdict_map = {
        "plausible": "primary_thesis",
        "implausible": "counter_thesis",
        "uncertain": "undecided",
    }
    payload["winning_side"] = legacy_verdict_map.get(aqa_verdict, "undecided")
    payload["winning_side_canonical"] = canonical_verdict_map.get(
        aqa_verdict, "undecided"
    )
    payload["confidence"] = abs(float(aqa_net.get("final", 0.0)))
    repaired_aspic_files = _persist_repaired_aspic_files(claim, payload)
    if repaired_aspic_files:
        payload["repaired_aspic_files"] = repaired_aspic_files
        _emit_progress(
            "evaluation_partial", {"repaired_aspic_files": repaired_aspic_files}
        )
    aqa_report_file = _persist_aqa_report_file(claim, payload)
    if aqa_report_file:
        payload["aqa_report_file"] = aqa_report_file
        _emit_progress("evaluation_partial", {"aqa_report_file": aqa_report_file})
    _emit_progress("evaluation_result", payload)
    _emit_phase("final_evaluation", "done", 100, "Valutazione completata")
    return payload


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Endpoint per la valutazione finale.
    Verifica la consistenza delle argomentazioni con la knowledge base via Neo4j.
    """
    try:
        data = request.get_json(silent=True) or {}
        payload = _run_evaluate_only(data)
        return jsonify(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Errore evaluation: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/stream", methods=["POST"])
def evaluate_stream():
    """Endpoint SSE con streaming live del solo Polisher-Evaluator."""
    if not _pipeline_lock.acquire(blocking=False):
        return jsonify({"error": "A pipeline is already running. Please wait."}), 429

    data = request.get_json(silent=True) or {}
    event_queue: Queue = Queue()
    sentinel = object()
    run_id = uuid.uuid4().hex
    cancel_event = threading.Event()

    global _active_stream_run
    with _active_stream_run_lock:
        _active_stream_run = {
            "run_id": run_id,
            "cancel_event": cancel_event,
            "started_at": time.time(),
            "kind": "evaluate_stream",
        }

    def push_progress(event_name: str, payload: dict) -> None:
        if cancel_event.is_set():
            raise PipelineCancelled("Esecuzione interrotta manualmente.")
        event_queue.put((event_name, payload))

    def push_error(message: str, code: int) -> None:
        event_queue.put(("error", {"message": message, "code": code}))

    def _worker() -> None:
        global _active_stream_run
        try:
            result = _run_evaluate_only(
                data,
                progress_callback=push_progress,
                cancel_event=cancel_event,
            )
            event_queue.put(("final", result))
        except PipelineCancelled as e:
            event_queue.put(
                ("cancelled", {"message": str(e), "run_id": run_id, "ok": True})
            )
        except ValueError as e:
            push_error(str(e), 400)
        except Exception as e:
            print(f"\n{'='*70}")
            print("❌ EVALUATE STREAM ERROR")
            print(f"{'='*70}")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            push_error(str(e), 500)
        finally:
            event_queue.put(("done", {"ok": True}))
            event_queue.put(sentinel)
            with _active_stream_run_lock:
                if (
                    isinstance(_active_stream_run, dict)
                    and _active_stream_run.get("run_id") == run_id
                ):
                    _active_stream_run = None
            _pipeline_lock.release()

    threading.Thread(target=_worker, daemon=True).start()
    event_queue.put(("run_started", {"run_id": run_id}))

    @stream_with_context
    def generate():
        while True:
            try:
                item = event_queue.get(timeout=12)
            except Empty:
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


@app.route("/api/doe/log", methods=["POST"])
def persist_doe_log():
    """Persist one consolidated DoE experiment log/report."""
    try:
        data = request.get_json(silent=True) or {}
        claim = (data.get("claim", "") or "").strip()
        if not claim:
            raise ValueError('Campo "claim" obbligatorio')
        artifacts = _persist_doe_experiment_files(claim, data)
        return jsonify(artifacts)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Errore persistenza log DoE: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/pdf/export", methods=["POST"])
def persist_pdf_export():
    """Persist one exported PDF artifact from the frontend."""
    try:
        pdf_file = request.files.get("pdf")
        if pdf_file is None:
            raise ValueError('Campo file "pdf" obbligatorio')

        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            raise ValueError("PDF vuoto")

        claim = (request.form.get("claim", "") or "").strip()
        prefix = (request.form.get("prefix", "") or "pipeline").strip()
        export_context = (request.form.get("export_context", "") or "pipeline").strip()
        client_filename = (request.form.get("client_filename", "") or "").strip()

        artifact = _persist_pdf_export_file(
            claim,
            pdf_bytes,
            export_context=export_context,
            prefix=prefix,
            client_filename=client_filename,
        )
        if not artifact:
            raise RuntimeError("Salvataggio PDF non riuscito")

        return jsonify({"pdf_file": artifact})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Errore persistenza PDF export: {e}")
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
    print("  • POST /api/counter_reason/stream - Contro-ragionamento (SSE)")
    print("  • POST /api/pipeline        - Pipeline completa")
    print("  • POST /api/pipeline/stream - Pipeline completa (SSE token streaming)")
    print("  • POST /api/pipeline/stop   - Interrompe pipeline SSE attiva")
    print("  • POST /api/evaluate        - Valutazione finale (stub)")
    print("  • POST /api/evaluate/stream - Valutazione finale (SSE)")
    print("  • POST /api/doe/log         - Persistenza log/report DoE")
    print("  • POST /api/pdf/export      - Persistenza PDF esportato")
    print()
    print("=" * 70)
    print()

    app.run(host=settings.api_host, port=settings.api_port, debug=settings.debug)
