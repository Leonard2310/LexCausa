#!/usr/bin/env python3
"""
Domain-specific retrieval tuning from claims.md.

Modes:
- Supervised (preferred): optimize on gold labels with Hit@k, MRR, nDCG@k.
- Unsupervised fallback: proxy lexical metrics when gold labels are missing.

Inputs:
- claims.md (covered claims sections)
- claims_gold_labels.json (claim_id -> expected articles)

Outputs:
- logs/tuning/retrieval_tuning_YYYYMMDD_HHMMSS.json
- logs/tuning/retrieval_tuning_latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import settings  # noqa: E402
from services.legal_search import LegalSearchPipeline  # noqa: E402

CLAIMS_MD_PATH = PROJECT_ROOT / "claims.md"
GOLD_LABELS_PATH = PROJECT_ROOT / "claims_gold_labels.json"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "tuning"
DEFAULT_TOP_K = 30


@dataclass
class ClaimEntry:
    claim_id: str
    section: str
    domain: str
    title: str
    text: str


@dataclass
class EvalEntry:
    claim_id: str
    domain: str
    source: str
    proxy_score: float
    mean_overlap_top10: float
    coverage_terms_top10: float
    top_sources: dict[str, int]
    top_articles: list[str]
    gold_count: int = 0
    hit_at_5: Optional[float] = None
    hit_at_10: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_10: Optional[float] = None


def _iter_progress(iterable, enabled: bool, **kwargs):
    """Return tqdm wrapper when enabled, otherwise plain iterable."""
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


def _extract_block(text: str, start_marker: str, end_markers: list[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def _parse_claims_from_block(
    block: str,
    pattern: re.Pattern,
    section: str,
    domain: str,
) -> list[ClaimEntry]:
    claims: list[ClaimEntry] = []
    matches = list(pattern.finditer(block))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        body = block[start:end].strip()
        if not body:
            continue
        claim_id = str(match.group(1)).strip().upper()
        title = str(match.group(2)).strip()
        claims.append(
            ClaimEntry(
                claim_id=claim_id,
                section=section,
                domain=domain,
                title=title,
                text=body,
            )
        )
    return claims


def parse_claims_md(path: Path) -> tuple[list[ClaimEntry], int]:
    text = path.read_text(encoding="utf-8")

    civile_block = _extract_block(
        text,
        "## CLAIM CIVILI (COPERTI)",
        [
            "## CLAIM PENALI (COPERTI)",
            "## CLAIM MIXED (COPERTI)",
            "## CLAIM AMMINISTRATIVI (COPERTI)",
            "## CLAIM NON COPERTI",
        ],
    )
    penale_block = _extract_block(
        text,
        "## CLAIM PENALI (COPERTI)",
        [
            "## CLAIM MIXED (COPERTI)",
            "## CLAIM AMMINISTRATIVI (COPERTI)",
            "## CLAIM NON COPERTI",
        ],
    )
    mixed_block = _extract_block(
        text,
        "## CLAIM MIXED (COPERTI)",
        ["## CLAIM AMMINISTRATIVI (COPERTI)", "## CLAIM NON COPERTI"],
    )
    admin_block = _extract_block(
        text,
        "## CLAIM AMMINISTRATIVI (COPERTI)",
        ["## CLAIM NON COPERTI"],
    )
    non_covered_block = _extract_block(text, "## CLAIM NON COPERTI", [])

    claim_pattern = re.compile(r"(?m)^\s*###\s*([A-Z]\d+)\s*-\s*(.+?)\s*$")

    civile_claims = _parse_claims_from_block(
        civile_block,
        claim_pattern,
        section="CLAIM CIVILI (COPERTI)",
        domain="civile",
    )
    penale_claims = _parse_claims_from_block(
        penale_block,
        claim_pattern,
        section="CLAIM PENALI (COPERTI)",
        domain="penale",
    )
    mixed_claims = _parse_claims_from_block(
        mixed_block,
        claim_pattern,
        section="CLAIM MIXED (COPERTI)",
        domain="misto",
    )
    admin_claims = _parse_claims_from_block(
        admin_block,
        claim_pattern,
        section="CLAIM AMMINISTRATIVI (COPERTI)",
        domain="amministrativo",
    )

    all_claims = civile_claims + penale_claims + mixed_claims + admin_claims
    non_covered_count = len(
        re.findall(r"(?m)^\s*###\s*NC\d+\s*-\s*.+$", non_covered_block)
    )
    return all_claims, non_covered_count


def _normalize_source(value: str) -> str:
    raw = (value or "").strip().lower().replace(" ", "")
    mapping = {
        "codice_penale": "codice_penale",
        "cp": "codice_penale",
        "c.p.": "codice_penale",
        "codice_civile": "codice_civile",
        "cc": "codice_civile",
        "c.c.": "codice_civile",
        "codice_amministrativo": "codice_amministrativo",
        "amm": "codice_amministrativo",
        "l241": "codice_amministrativo",
        "l.241/1990": "codice_amministrativo",
    }
    return mapping.get(raw, raw)


def _normalize_articolo(value: str) -> str:
    art = (value or "").strip().lower()
    art = art.replace("–", "-").replace("—", "-")
    art = re.sub(r"^art\.?\s*", "", art)
    art = re.sub(r"\s+", "", art)
    art = art.strip(". ,;:")
    return art


def _normalize_articolo_for_source(source: str, value: str) -> str:
    art = _normalize_articolo(value)
    src = _normalize_source(source)
    # codice_civile_normattiva uses article_id style: art1, art2, ...
    if src == "codice_civile" and art and not art.startswith("art"):
        art = f"art{art}"
    return art


def _article_key(source: str, articolo: str) -> str:
    src = _normalize_source(source)
    art = _normalize_articolo_for_source(src, articolo)
    return f"{src}:art.{art}"


def load_gold_labels(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels_raw = payload.get("labels", payload)
    if not isinstance(labels_raw, dict):
        return {}

    labels: dict[str, set[str]] = {}
    for claim_id, entries in labels_raw.items():
        cid = str(claim_id).strip().upper()
        normalized: set[str] = set()

        if not isinstance(entries, list):
            continue

        for item in entries:
            source = ""
            articolo = ""
            if isinstance(item, dict):
                source = str(item.get("source", "")).strip()
                articolo = str(item.get("articolo", "")).strip()
            elif isinstance(item, str):
                text = item.strip()
                if ":" in text:
                    left, right = text.split(":", 1)
                    source = left.strip()
                    articolo = right.replace("art.", "").strip()
                else:
                    articolo = text
            if not source or not articolo:
                continue
            normalized.add(_article_key(source, articolo))

        if normalized:
            labels[cid] = normalized

    return labels


def _mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _score_claim_proxy(
    pipeline: LegalSearchPipeline,
    claim_text: str,
    results,
) -> tuple[float, float, float]:
    terms = pipeline.get_search_query_terms(claim_text, mode="llm")
    top = results[:10]
    if not top:
        return 0.0, 0.0, 0.0
    if not terms:
        top1 = max(0.0, float(top[0].score))
        top5 = float(top[min(len(top) - 1, 4)].score)
        margin = max(0.0, top1 - top5)
        return max(0.0, min(1.0, margin)), 0.0, 0.0

    covered_terms: set[str] = set()
    weighted_overlap = 0.0
    weight_sum = 0.0
    overlap_values: list[float] = []

    for rank, item in enumerate(top, start=1):
        text = f"{item.articolo} {item.titolo} {item.testo[:600]}".lower()
        item_terms = set(re.findall(r"[a-zA-Zàèéìòù0-9\\-]+", text))
        overlap = len(terms & item_terms) / max(1, len(terms))
        overlap_values.append(overlap)
        covered_terms.update(terms & item_terms)
        weight = 1.0 / rank
        weighted_overlap += overlap * weight
        weight_sum += weight

    top1 = max(0.0, float(top[0].score))
    top5 = float(top[min(len(top) - 1, 4)].score)
    margin = max(0.0, top1 - top5)
    weighted_overlap = weighted_overlap / max(1e-9, weight_sum)
    coverage_ratio = len(covered_terms) / max(1, len(terms))
    mean_overlap = statistics.fmean(overlap_values) if overlap_values else 0.0
    proxy_score = (0.55 * weighted_overlap) + (0.35 * coverage_ratio) + (0.10 * margin)
    proxy_score = max(0.0, min(1.0, proxy_score))
    return proxy_score, mean_overlap, coverage_ratio


def _ranked_article_keys(results) -> list[str]:
    return [_article_key(item.source, item.articolo) for item in results]


def _hit_at_k(ranked_keys: list[str], gold_keys: set[str], k: int) -> float:
    if not ranked_keys or not gold_keys or k <= 0:
        return 0.0
    return 1.0 if any(key in gold_keys for key in ranked_keys[:k]) else 0.0


def _mrr(ranked_keys: list[str], gold_keys: set[str]) -> float:
    if not ranked_keys or not gold_keys:
        return 0.0
    for idx, key in enumerate(ranked_keys, start=1):
        if key in gold_keys:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(ranked_keys: list[str], gold_keys: set[str], k: int) -> float:
    if not ranked_keys or not gold_keys or k <= 0:
        return 0.0
    dcg = 0.0
    for idx, key in enumerate(ranked_keys[:k], start=1):
        if key in gold_keys:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_keys), k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _supervised_metrics(
    ranked_keys: list[str],
    gold_keys: set[str],
) -> dict[str, float]:
    hit5 = _hit_at_k(ranked_keys, gold_keys, 5)
    hit10 = _hit_at_k(ranked_keys, gold_keys, 10)
    mrr = _mrr(ranked_keys, gold_keys)
    ndcg10 = _ndcg_at_k(ranked_keys, gold_keys, 10)
    return {
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "mrr": mrr,
        "ndcg_at_10": ndcg10,
    }


def _supervised_objective(metrics: dict[str, float]) -> float:
    # Weighted objective to optimize ranking quality.
    return (
        (0.15 * metrics["hit_at_5"])
        + (0.15 * metrics["hit_at_10"])
        + (0.30 * metrics["mrr"])
        + (0.40 * metrics["ndcg_at_10"])
    )


def _domain_keys(domain: str) -> dict[str, str]:
    if domain == "penale":
        return {
            "vector_weight": "search_hybrid_penale_vector_weight",
            "fulltext_weight": "search_hybrid_penale_fulltext_weight",
            "keyword_min_overlap_count": "search_hybrid_penale_keyword_min_overlap_count",
            "zero_overlap_multiplier": "search_hybrid_penale_zero_overlap_multiplier",
            "low_overlap_multiplier": "search_hybrid_penale_low_overlap_multiplier",
        }
    if domain == "civile":
        return {
            "vector_weight": "search_hybrid_civile_vector_weight",
            "fulltext_weight": "search_hybrid_civile_fulltext_weight",
            "keyword_min_overlap_count": "search_hybrid_civile_keyword_min_overlap_count",
            "zero_overlap_multiplier": "search_hybrid_civile_zero_overlap_multiplier",
            "low_overlap_multiplier": "search_hybrid_civile_low_overlap_multiplier",
        }
    if domain == "amministrativo":
        return {
            "vector_weight": "search_hybrid_admin_vector_weight",
            "fulltext_weight": "search_hybrid_admin_fulltext_weight",
            "keyword_min_overlap_count": "search_hybrid_admin_keyword_min_overlap_count",
            "zero_overlap_multiplier": "search_hybrid_admin_zero_overlap_multiplier",
            "low_overlap_multiplier": "search_hybrid_admin_low_overlap_multiplier",
        }
    raise ValueError(f"Unsupported domain: {domain}")


def _source_for_domain(domain: str) -> str:
    return {
        "penale": "codice_penale",
        "civile": "codice_civile",
        "amministrativo": "codice_amministrativo",
    }[domain]


def _candidate_grid_for_domain(domain: str) -> list[dict[str, Any]]:
    if domain == "penale":
        vector_weights = [0.45, 0.55, 0.60, 0.65, 0.75]
        min_overlaps = [1, 2, 3]
        zero_mults = [0.75, 0.85, 0.95, 1.0]
        low_mults = [0.88, 0.94, 1.0]
    elif domain == "civile":
        vector_weights = [0.20, 0.30, 0.40, 0.50, 0.60]
        min_overlaps = [1, 2, 3]
        zero_mults = [0.80, 0.90, 1.0]
        low_mults = [0.90, 0.95, 1.0]
    else:  # amministrativo
        vector_weights = [0.20, 0.35, 0.50]
        min_overlaps = [1, 2, 3]
        zero_mults = [0.85, 0.95, 1.0]
        low_mults = [0.95, 1.0]

    candidates: list[dict[str, Any]] = []
    for vw in vector_weights:
        fw = round(1.0 - vw, 2)
        for mo in min_overlaps:
            for zm in zero_mults:
                for lm in low_mults:
                    if lm < zm:
                        continue
                    candidates.append(
                        {
                            "vector_weight": float(vw),
                            "fulltext_weight": float(fw),
                            "keyword_min_overlap_count": int(mo),
                            "zero_overlap_multiplier": float(zm),
                            "low_overlap_multiplier": float(lm),
                        }
                    )
    return candidates


def _get_current_params(domain: str) -> dict[str, Any]:
    keys = _domain_keys(domain)
    return {k: getattr(settings, attr) for k, attr in keys.items()}


def _apply_params(domain: str, params: dict[str, Any]) -> None:
    keys = _domain_keys(domain)
    for p_name, attr in keys.items():
        setattr(settings, attr, params[p_name])


def _precompute_ranked_lists(
    pipeline: LegalSearchPipeline,
    claims: list[ClaimEntry],
    source: str,
    top_k: int,
    query_terms_mode: str,
    show_progress: bool = True,
) -> dict[str, dict[str, Any]]:
    precomputed: dict[str, dict[str, Any]] = {}
    candidate_limit = max(
        top_k * settings.search_hybrid_candidate_multiplier,
        settings.search_hybrid_candidate_min,
    )
    fused_limit = max(top_k * settings.search_hybrid_fused_pool_multiplier, top_k)

    with pipeline.driver.session() as session:
        for claim in _iter_progress(
            claims,
            enabled=show_progress,
            desc=f"[{source}] precompute",
            unit="claim",
            leave=False,
        ):
            embedding = pipeline.embed_text(claim.text)
            query_text = claim.text.strip()
            query_terms = pipeline.get_search_query_terms(
                query_text, mode=query_terms_mode
            )
            fulltext_query = " ".join(sorted(query_terms)) if query_terms else ""
            vector_results = pipeline._vector_exact_search(
                session=session,
                embedding=embedding,
                source=source,
                libro="",
                limit=candidate_limit,
            )
            fulltext_results = pipeline._fulltext_search(
                session=session,
                query_text=fulltext_query,
                source=source,
                libro="",
                limit=candidate_limit,
            )
            precomputed[claim.claim_id] = {
                "claim": claim,
                "query_text": query_text,
                "vector_results": vector_results,
                "fulltext_results": fulltext_results,
                "fused_limit": fused_limit,
            }
    return precomputed


def tune_domain(
    pipeline: LegalSearchPipeline,
    domain: str,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    supervised: bool,
    query_terms_mode: str,
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = True,
) -> dict[str, Any]:
    base_params = _get_current_params(domain)
    if not claims:
        return {
            "domain": domain,
            "claims_count": 0,
            "labeled_claims_count": 0,
            "status": "skipped_no_covered_claims",
            "best_params": base_params,
            "best_score": None,
            "num_candidates": 0,
            "tuning_mode": "n/a",
            "query_terms_mode": query_terms_mode,
        }

    labeled_claim_ids = {
        c.claim_id
        for c in claims
        if c.claim_id in gold_labels and gold_labels[c.claim_id]
    }
    use_supervised = supervised and len(labeled_claim_ids) > 0

    source = _source_for_domain(domain)
    precomputed = _precompute_ranked_lists(
        pipeline,
        claims,
        source,
        top_k=top_k,
        query_terms_mode=query_terms_mode,
        show_progress=show_progress,
    )
    candidates = _candidate_grid_for_domain(domain)

    best_score = -math.inf
    best_params = dict(base_params)

    for candidate in _iter_progress(
        candidates,
        enabled=show_progress,
        desc=f"[{domain}] grid",
        unit="cand",
        leave=False,
    ):
        _apply_params(domain, candidate)
        claim_scores: list[float] = []

        for entry in precomputed.values():
            claim = entry["claim"]
            fused = pipeline._fuse_ranked_results(
                source=source,
                vector_results=entry["vector_results"],
                fulltext_results=entry["fulltext_results"],
                query_text=entry["query_text"],
                limit=entry["fused_limit"],
            )[:top_k]

            if use_supervised:
                gold = gold_labels.get(claim.claim_id, set())
                if not gold:
                    continue
                metrics = _supervised_metrics(_ranked_article_keys(fused), gold)
                claim_scores.append(_supervised_objective(metrics))
            else:
                proxy, _, _ = _score_claim_proxy(pipeline, claim.text, fused)
                claim_scores.append(proxy)

        score = statistics.fmean(claim_scores) if claim_scores else 0.0
        if score > best_score:
            best_score = score
            best_params = dict(candidate)

    _apply_params(domain, best_params)

    return {
        "domain": domain,
        "claims_count": len(claims),
        "labeled_claims_count": len(labeled_claim_ids),
        "status": "ok",
        "best_params": best_params,
        "best_score": round(float(best_score), 6),
        "num_candidates": len(candidates),
        "tuning_mode": "supervised" if use_supervised else "proxy_fallback",
        "query_terms_mode": query_terms_mode,
    }


def evaluate_domain(
    pipeline: LegalSearchPipeline,
    domain: str,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    query_terms_mode: str,
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = True,
) -> dict[str, Any]:
    if not claims:
        return {
            "domain": domain,
            "claims_count": 0,
            "labeled_claims_count": 0,
            "status": "skipped_no_covered_claims",
            "mean_proxy_score": None,
            "mean_overlap_top10": None,
            "mean_coverage_terms_top10": None,
            "mean_hit_at_5": None,
            "mean_hit_at_10": None,
            "mean_mrr": None,
            "mean_ndcg_at_10": None,
            "entries": [],
        }

    source = _source_for_domain(domain)
    filters = [(source, "")]
    entries: list[EvalEntry] = []

    hit5_vals: list[float] = []
    hit10_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []

    for claim in _iter_progress(
        claims,
        enabled=show_progress,
        desc=f"[{domain}] benchmark",
        unit="claim",
        leave=False,
    ):
        embedding = pipeline.embed_text(claim.text)
        results = pipeline.vector_search(
            embedding=embedding,
            libri_filters=filters,
            top_k=top_k,
            query_text=claim.text,
        )
        results = pipeline.expand_with_cited_articles(results)[:top_k]

        proxy, mean_overlap, coverage_terms = _score_claim_proxy(
            pipeline, claim.text, results
        )
        top_sources: dict[str, int] = {}
        top_articles = []
        for item in results[:10]:
            top_sources[item.source] = top_sources.get(item.source, 0) + 1
            top_articles.append(_article_key(item.source, item.articolo))

        gold = gold_labels.get(claim.claim_id, set())
        metrics = (
            _supervised_metrics(_ranked_article_keys(results), gold) if gold else None
        )
        if metrics:
            hit5_vals.append(metrics["hit_at_5"])
            hit10_vals.append(metrics["hit_at_10"])
            mrr_vals.append(metrics["mrr"])
            ndcg_vals.append(metrics["ndcg_at_10"])

        entries.append(
            EvalEntry(
                claim_id=claim.claim_id,
                domain=domain,
                source=source,
                proxy_score=float(proxy),
                mean_overlap_top10=float(mean_overlap),
                coverage_terms_top10=float(coverage_terms),
                top_sources=top_sources,
                top_articles=top_articles,
                gold_count=len(gold),
                hit_at_5=(metrics["hit_at_5"] if metrics else None),
                hit_at_10=(metrics["hit_at_10"] if metrics else None),
                mrr=(metrics["mrr"] if metrics else None),
                ndcg_at_10=(metrics["ndcg_at_10"] if metrics else None),
            )
        )

    mean_hit5 = _mean_or_none(hit5_vals)
    mean_hit10 = _mean_or_none(hit10_vals)
    mean_mrr = _mean_or_none(mrr_vals)
    mean_ndcg = _mean_or_none(ndcg_vals)

    return {
        "domain": domain,
        "claims_count": len(claims),
        "labeled_claims_count": len([c for c in claims if gold_labels.get(c.claim_id)]),
        "status": "ok",
        "query_terms_mode": query_terms_mode,
        "mean_proxy_score": round(statistics.fmean(e.proxy_score for e in entries), 6),
        "mean_overlap_top10": round(
            statistics.fmean(e.mean_overlap_top10 for e in entries), 6
        ),
        "mean_coverage_terms_top10": round(
            statistics.fmean(e.coverage_terms_top10 for e in entries), 6
        ),
        "mean_hit_at_5": (round(mean_hit5, 6) if mean_hit5 is not None else None),
        "mean_hit_at_10": (round(mean_hit10, 6) if mean_hit10 is not None else None),
        "mean_mrr": (round(mean_mrr, 6) if mean_mrr is not None else None),
        "mean_ndcg_at_10": (round(mean_ndcg, 6) if mean_ndcg is not None else None),
        "entries": [asdict(e) for e in entries],
    }


def evaluate_mixed(
    pipeline: LegalSearchPipeline,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    query_terms_mode: str,
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = True,
) -> dict[str, Any]:
    if not claims:
        return {
            "domain": "misto",
            "claims_count": 0,
            "labeled_claims_count": 0,
            "status": "skipped_no_covered_claims",
            "mean_proxy_score": None,
            "mean_hit_at_5": None,
            "mean_hit_at_10": None,
            "mean_mrr": None,
            "mean_ndcg_at_10": None,
            "entries": [],
        }

    filters = [
        ("codice_civile", ""),
        ("codice_penale", ""),
        ("codice_amministrativo", ""),
    ]

    entries: list[EvalEntry] = []
    hit5_vals: list[float] = []
    hit10_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []

    for claim in _iter_progress(
        claims,
        enabled=show_progress,
        desc="[misto] benchmark",
        unit="claim",
        leave=False,
    ):
        embedding = pipeline.embed_text(claim.text)
        results = pipeline.vector_search(
            embedding=embedding,
            libri_filters=filters,
            top_k=top_k,
            query_text=claim.text,
        )
        results = pipeline.expand_with_cited_articles(results)[:top_k]

        proxy, mean_overlap, coverage_terms = _score_claim_proxy(
            pipeline, claim.text, results
        )
        top_sources: dict[str, int] = {}
        top_articles = []
        for item in results[:10]:
            top_sources[item.source] = top_sources.get(item.source, 0) + 1
            top_articles.append(_article_key(item.source, item.articolo))

        gold = gold_labels.get(claim.claim_id, set())
        metrics = (
            _supervised_metrics(_ranked_article_keys(results), gold) if gold else None
        )
        if metrics:
            hit5_vals.append(metrics["hit_at_5"])
            hit10_vals.append(metrics["hit_at_10"])
            mrr_vals.append(metrics["mrr"])
            ndcg_vals.append(metrics["ndcg_at_10"])

        entries.append(
            EvalEntry(
                claim_id=claim.claim_id,
                domain="misto",
                source="multi",
                proxy_score=float(proxy),
                mean_overlap_top10=float(mean_overlap),
                coverage_terms_top10=float(coverage_terms),
                top_sources=top_sources,
                top_articles=top_articles,
                gold_count=len(gold),
                hit_at_5=(metrics["hit_at_5"] if metrics else None),
                hit_at_10=(metrics["hit_at_10"] if metrics else None),
                mrr=(metrics["mrr"] if metrics else None),
                ndcg_at_10=(metrics["ndcg_at_10"] if metrics else None),
            )
        )

    mean_hit5 = _mean_or_none(hit5_vals)
    mean_hit10 = _mean_or_none(hit10_vals)
    mean_mrr = _mean_or_none(mrr_vals)
    mean_ndcg = _mean_or_none(ndcg_vals)

    return {
        "domain": "misto",
        "claims_count": len(claims),
        "labeled_claims_count": len([c for c in claims if gold_labels.get(c.claim_id)]),
        "status": "ok",
        "query_terms_mode": query_terms_mode,
        "mean_proxy_score": round(statistics.fmean(e.proxy_score for e in entries), 6),
        "mean_overlap_top10": round(
            statistics.fmean(e.mean_overlap_top10 for e in entries), 6
        ),
        "mean_coverage_terms_top10": round(
            statistics.fmean(e.coverage_terms_top10 for e in entries), 6
        ),
        "mean_hit_at_5": (round(mean_hit5, 6) if mean_hit5 is not None else None),
        "mean_hit_at_10": (round(mean_hit10, 6) if mean_hit10 is not None else None),
        "mean_mrr": (round(mean_mrr, 6) if mean_mrr is not None else None),
        "mean_ndcg_at_10": (round(mean_ndcg, 6) if mean_ndcg is not None else None),
        "entries": [asdict(e) for e in entries],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune retrieval parameters on claims set."
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--gold-labels",
        type=Path,
        default=GOLD_LABELS_PATH,
        help="Path to claims gold labels JSON.",
    )
    parser.add_argument(
        "--query-terms-mode",
        choices=["llm"],
        default="llm",
        help="Term extraction mode used to build fulltext query (LLM-only).",
    )
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        help="Force proxy-only tuning even if gold labels are available.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    top_k = max(1, int(args.top_k))
    query_terms_mode = str(args.query_terms_mode).strip().lower()
    show_progress = not args.no_progress

    covered_claims, non_covered_count = parse_claims_md(CLAIMS_MD_PATH)
    gold_labels = load_gold_labels(args.gold_labels)

    by_domain: dict[str, list[ClaimEntry]] = {
        "penale": [c for c in covered_claims if c.domain == "penale"],
        "civile": [c for c in covered_claims if c.domain == "civile"],
        "amministrativo": [c for c in covered_claims if c.domain == "amministrativo"],
        "misto": [c for c in covered_claims if c.domain == "misto"],
    }

    supervised = (not args.unsupervised) and bool(gold_labels)

    print("=== Claims dataset ===")
    print(f"Totale claims coperti parsed: {len(covered_claims)}")
    print(f"Non coperti dichiarati: {non_covered_count}")
    print(f"Claims coperti usati: {len(covered_claims)}")
    print(f"Gold labels loaded: {len(gold_labels)} claim(s) from {args.gold_labels}")
    print(f"Tuning mode requested: {'supervised' if supervised else 'proxy_fallback'}")
    print(f"Query terms mode: {query_terms_mode}")
    for k in ("penale", "civile", "amministrativo", "misto"):
        print(f"- {k}: {len(by_domain[k])}")

    if not supervised:
        print("⚠️ Using proxy fallback (no gold labels or --unsupervised).")

    pipeline = LegalSearchPipeline()
    try:
        original_params = {
            "penale": _get_current_params("penale"),
            "civile": _get_current_params("civile"),
            "amministrativo": _get_current_params("amministrativo"),
        }

        tuning_summary = {
            "penale": tune_domain(
                pipeline,
                "penale",
                by_domain["penale"],
                gold_labels,
                supervised,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
            "civile": tune_domain(
                pipeline,
                "civile",
                by_domain["civile"],
                gold_labels,
                supervised,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
            "amministrativo": tune_domain(
                pipeline,
                "amministrativo",
                by_domain["amministrativo"],
                gold_labels,
                supervised,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
        }

        benchmark_summary = {
            "penale": evaluate_domain(
                pipeline,
                "penale",
                by_domain["penale"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
            "civile": evaluate_domain(
                pipeline,
                "civile",
                by_domain["civile"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
            "amministrativo": evaluate_domain(
                pipeline,
                "amministrativo",
                by_domain["amministrativo"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
            "misto": evaluate_mixed(
                pipeline,
                by_domain["misto"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
            ),
        }
    finally:
        pipeline.close()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "claims_path": str(CLAIMS_MD_PATH),
        "gold_labels_path": str(args.gold_labels),
        "query_terms_mode": query_terms_mode,
        "supervised_requested": not args.unsupervised,
        "supervised_effective": supervised,
        "claims_counts": {
            "parsed_total": len(covered_claims),
            "non_covered_excluded": non_covered_count,
            "covered_used": len(covered_claims),
            "labeled_claims": len(gold_labels),
            "by_domain": {k: len(v) for k, v in by_domain.items()},
        },
        "original_params": original_params,
        "tuning_summary": tuning_summary,
        "benchmark_summary": benchmark_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"retrieval_tuning_{ts}.json"
    latest_path = OUTPUT_DIR / "retrieval_tuning_latest.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Tuning summary ===")
    for domain in ("penale", "civile", "amministrativo"):
        item = tuning_summary[domain]
        print(
            f"- {domain}: status={item['status']}, claims={item['claims_count']}, "
            f"labeled={item['labeled_claims_count']}, "
            f"mode={item['tuning_mode']}, best_score={item['best_score']}"
        )
        print(f"  best_params={item['best_params']}")

    print("\n=== Benchmark summary ===")
    for domain in ("penale", "civile", "amministrativo", "misto"):
        item = benchmark_summary[domain]
        print(
            f"- {domain}: status={item['status']}, claims={item['claims_count']}, "
            f"labeled={item['labeled_claims_count']}, "
            f"proxy={item['mean_proxy_score']}, "
            f"hit@5={item.get('mean_hit_at_5')}, hit@10={item.get('mean_hit_at_10')}, "
            f"mrr={item.get('mean_mrr')}, ndcg@10={item.get('mean_ndcg_at_10')}"
        )

    print("\nSaved report:")
    print(f"- {out_path}")
    print(f"- {latest_path}")


if __name__ == "__main__":
    main()
