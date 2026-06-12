#!/usr/bin/env python
"""
Multi-DoE orchestrator for Ibisco HPC (vLLM offline inference).

Loads LLM models via vLLM directly (in-process, no HTTP server, no Groq Cloud),
calls the LexCausa pipeline Python modules, and saves metrics.csv ready for
analyze_multi_doe.py.

Full-factorial design: Reasoner model and Counter-Reasoner model vary
independently (RQ1), crossed with planning ablation (RQ3).  All responses are
generated in English (response_language='en') for consistency on HPC runs.

Usage (inside a SLURM job or interactive session with GPU):

    export HF_TOKEN=<your_token>

    python scripts/run_multi_doe_ibisco.py \\
        --seed 42 \\
        --replicates 10 \\
        --reasoner-models deepseek_r1,gpt_oss_120b,groq_llama_3_3_70b_versatile,qwen_25_72b \\
        --counter-models  deepseek_r1,gpt_oss_120b,groq_llama_3_3_70b_versatile,qwen_25_72b \\
        --domains CIVILE,PENALE,AMMINISTRATIVO,MISTO \\
        --planning-ablations on,off \\
        --causality-ablations on,off \\
        --out experiments/multi_doe/runs/ibisco_$(date +%Y%m%d_%H%M%S)

Seed strategy: replicate i uses seed = base_seed + i for reproducibility.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Ensure src/ is importable ─────────────────────────────────────────────────
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Set backend BEFORE importing anything from src/ so Settings picks it up.
os.environ.setdefault("LLM_BACKEND", "local")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


_DEFAULT_REASONING_MODELS = "deepseek_r1,gpt_oss_120b"
_DEFAULT_NON_REASONING_MODELS = "groq_llama_3_3_70b_versatile,qwen_25_72b"
_ALL_DEFAULT_MODELS = f"{_DEFAULT_REASONING_MODELS},{_DEFAULT_NON_REASONING_MODELS}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-DoE on Ibisco HPC (vLLM offline inference) — full factorial R×C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument(
        "--replicates", type=int, default=10, help="Replicates per condition"
    )
    p.add_argument(
        "--reasoner-models",
        default=_ALL_DEFAULT_MODELS,
        help="Comma-separated aliases for the Reasoner (RQ1)",
    )
    p.add_argument(
        "--counter-models",
        default=_ALL_DEFAULT_MODELS,
        help="Comma-separated aliases for the Counter-Reasoner (RQ1)",
    )
    p.add_argument(
        "--domains",
        default="CIVILE,PENALE,AMMINISTRATIVO,MISTO",
        help="Comma-separated legal domains",
    )
    p.add_argument(
        "--planning-ablations",
        default="on,off",
        choices=["on,off", "on", "off"],
        help="Planning ablation for counter-reasoner causality (RQ3)",
    )
    p.add_argument(
        "--causality-ablations",
        default="on",
        choices=["on,off", "on", "off"],
        help="Causality taxonomy ablation on counter (on/off or both)",
    )
    p.add_argument(
        "--claims-file", default="claims.md", help="Path to claims markdown file"
    )
    p.add_argument(
        "--fixed-model",
        default=None,
        help=(
            "Model alias for retrieval/AQA/classifier phases (all non R/C phases). "
            "Defaults to the first loaded model."
        ),
    )
    p.add_argument(
        "--evaluator-model",
        default=None,
        help=(
            "Model alias for the Polisher-Evaluator. "
            "Defaults to --fixed-model or first loaded model."
        ),
    )
    p.add_argument("--out", required=True, help="Output directory for results")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Claims loader (same format as run_multi_doe.py)
# ──────────────────────────────────────────────────────────────────────────────


def _load_claims(claims_file: Path) -> list[dict]:
    """Parse claims.md → list of {id, domain, title, text}."""
    _DOMAIN_MAP = {
        "CIVILI": "CIVILE",
        "PENALI": "PENALE",
        "AMMINISTRATIVI": "AMMINISTRATIVO",
        "MISTO": "MISTO",
    }
    claims: list[dict] = []
    current_domain = "MISTO"
    current_id: Optional[str] = None
    current_title: Optional[str] = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_id is None:
            return
        claims.append(
            {
                "id": current_id,
                "domain": current_domain,
                "title": current_title or "",
                "text": "\n".join(current_lines).strip(),
            }
        )

    for line in claims_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            header = line[3:].upper()
            for key, val in _DOMAIN_MAP.items():
                if key in header:
                    current_domain = val
                    break
        elif line.startswith("### "):
            _flush()
            current_id = line[4:].split(" - ")[0].strip()
            current_title = (
                line[4:].split(" - ", 1)[-1].strip() if " - " in line[4:] else ""
            )
            current_lines = []
        elif current_id is not None:
            current_lines.append(line)

    _flush()
    return claims


# ──────────────────────────────────────────────────────────────────────────────
# Run matrix
# ──────────────────────────────────────────────────────────────────────────────


def _generate_matrix(
    claims: list[dict],
    reasoner_models: list[str],
    counter_models: list[str],
    domains: list[str],
    planning_ablations: list[tuple[bool, bool]],
    causality_ablations: list[bool],
    replicates: int,
    base_seed: int,
) -> pd.DataFrame:
    """Full-factorial: reasoner_model × counter_model × planning × causality × claims × replicates."""
    rows = []
    for replicate in range(replicates):
        seed = base_seed + replicate
        for r_model in reasoner_models:
            for c_model in counter_models:
                for plan_r, plan_c in planning_ablations:
                    for caus in causality_ablations:
                        for claim in claims:
                            if claim["domain"] not in domains:
                                continue
                            rows.append(
                                {
                                    "run_id": str(uuid.uuid4())[:8],
                                    "replicate": replicate,
                                    "seed": seed,
                                    "reasoner_model": r_model,
                                    "counter_model": c_model,
                                    "domain": claim["domain"],
                                    "claim_id": claim["id"],
                                    "claim_text": claim["text"],
                                    "planning_reasoner": plan_r,
                                    "planning_counter": plan_c,
                                    "causality_enabled": caus,
                                }
                            )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics extraction (mirrors run_multi_doe.py._extract_metrics_from_response)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_metrics(run_id: str, row: dict, response: dict, duration: float) -> dict:
    metrics: dict = {
        "run_id": run_id,
        "claim_id": row["claim_id"],
        "domain": row["domain"],
        "reasoner_model": row["reasoner_model"],
        "counter_model": row["counter_model"],
        "planning_reasoner": row["planning_reasoner"],
        "planning_counter": row["planning_counter"],
        "causality_enabled": bool(row.get("causality_enabled", True)),
        "replicate": row["replicate"],
        "seed": row["seed"],
        "duration_sec": round(duration, 2),
    }

    aqa_report = response.get("evaluation", {}).get("aqa_report", {})
    metrics["aqa_verdict"] = aqa_report.get("verdict", "unknown")
    net_plaus = aqa_report.get("net_plausibility", {})
    metrics["aqa_plausibility"] = float(net_plaus.get("final", 0.0))
    metrics["aqa_pro"] = float(net_plaus.get("pro", 0.0))
    metrics["aqa_contra"] = float(net_plaus.get("contra", 0.0))
    scores = aqa_report.get("scores", {})
    metrics["aqa_cogency"] = float(scores.get("cogency", 0.0))
    metrics["aqa_norm_support"] = float(scores.get("norm_support", 0.0))
    metrics["aqa_semantics"] = float(scores.get("semantics", 0.0))
    metrics["aqa_attack_coverage"] = float(aqa_report.get("attack_coverage_score", 0.0))

    consistency = response.get("evaluation", {}).get("consistency_report", {})
    ctr = consistency.get("counter_reasoner", {})
    metrics["citation_total"] = int(ctr.get("total_citations", 0))
    metrics["citation_valid"] = int(ctr.get("valid_citations", 0))
    metrics["citation_repaired"] = int(ctr.get("repaired_citations", 0))
    metrics["citation_dropped"] = int(ctr.get("dropped_citations", 0))
    metrics["citation_accuracy"] = metrics["citation_valid"] / max(
        1, metrics["citation_total"]
    )
    reas_c = consistency.get("reasoner", {})
    metrics["reasoner_citation_total"] = int(reas_c.get("total_citations", 0))
    metrics["reasoner_citation_valid"] = int(reas_c.get("valid_citations", 0))
    metrics["reasoner_citation_accuracy"] = metrics["reasoner_citation_valid"] / max(
        1, metrics["reasoner_citation_total"]
    )

    token_stats = response.get("_token_stats", {})
    metrics["total_tokens"] = int(token_stats.get("total_completion_tokens", 0))
    metrics["reasoning_tokens"] = int(token_stats.get("reasoning_completion_tokens", 0))
    metrics["counter_tokens"] = int(token_stats.get("counter_completion_tokens", 0))
    metrics["max_prompt_tokens_per_call"] = int(
        token_stats.get("max_prompt_tokens_per_call", 0)
    )

    reasoner = response.get("reasoner", {})
    metrics["reasoning_steps"] = len(reasoner.get("reasoning_chain", []))
    counter = response.get("counter_reasoner", {})
    metrics["counter_steps"] = len(counter.get("reasoning_chain", []))
    metrics["counter_attacks_count"] = len(counter.get("selected_attack_ids", []))

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    # ── Propagate env vars so Settings picks them up ──────────────────────────
    os.environ["LLM_BACKEND"] = "local"

    # ── Parse DoE parameters ──────────────────────────────────────────────────
    reasoner_models = [m.strip() for m in args.reasoner_models.split(",") if m.strip()]
    counter_models = [m.strip() for m in args.counter_models.split(",") if m.strip()]
    domains = [d.strip().upper() for d in args.domains.split(",")]
    planning_ablations: list[tuple[bool, bool]] = (
        [(True, True), (False, False)]
        if args.planning_ablations == "on,off"
        else [(True, True)] if args.planning_ablations == "on" else [(False, False)]
    )
    causality_ablations: list[bool] = (
        [True, False]
        if args.causality_ablations == "on,off"
        else [True] if args.causality_ablations == "on" else [False]
    )

    # ── Output directories ────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    log_path = out_dir / "logs" / "run_errors.log"

    # ── Import pipeline (after env vars are set) ──────────────────────────────
    from api_server import _run_full_pipeline  # noqa: E402
    from config import settings
    from services.vllm_client import (
        load_models,
        set_alias_map,
        set_default_model,
        set_reasoning_aliases,
        unload_model,
    )

    set_alias_map(settings.VLLM_ALIAS_MAP)
    set_reasoning_aliases(settings.VLLM_REASONING_ALIASES)

    # ── Load claims ───────────────────────────────────────────────────────────
    claims = _load_claims(Path(args.claims_file))
    print(f"✅ Loaded {len(claims)} claims")

    # ── Generate full-factorial run matrix ────────────────────────────────────
    matrix = _generate_matrix(
        claims,
        reasoner_models,
        counter_models,
        domains,
        planning_ablations,
        causality_ablations,
        args.replicates,
        args.seed,
    )
    # Sort by (reasoner_model, counter_model) so all conditions for a pair run
    # contiguously — minimises expensive vLLM load/unload cycles on HPC.
    matrix = matrix.sort_values(
        ["reasoner_model", "counter_model"], kind="stable"
    ).reset_index(drop=True)
    matrix.to_csv(out_dir / "run_matrix.csv", index=False)

    fixed_model = args.fixed_model
    evaluator_model = args.evaluator_model
    print(f"📋 Run matrix: {len(matrix)} runs")
    print(f"   Reasoner models:  {reasoner_models}")
    print(f"   Counter models:   {counter_models}")
    print(f"   Fixed model (support phases): {fixed_model or '(first loaded model)'}")
    print(
        f"   Evaluator model:  {evaluator_model or fixed_model or '(first loaded model)'}"
    )
    print(
        f"   Replicates: {args.replicates} (seeds {args.seed}–{args.seed + args.replicates - 1})"
    )
    print(f"   Output: {out_dir}")

    # Load fixed/support and evaluator models once upfront (stay loaded for the entire run).
    persistent_models = {m for m in (fixed_model, evaluator_model) if m}
    if persistent_models:
        load_models(list(persistent_models))
    if fixed_model:
        set_default_model(fixed_model)
        print(f"✅ Fixed model '{fixed_model}' loaded as default for support phases")
    if evaluator_model and evaluator_model != fixed_model:
        print(f"✅ Evaluator model '{evaluator_model}' loaded (persistent)")

    # ── Execute runs ──────────────────────────────────────────────────────────
    all_metrics: list[dict] = []
    started_at = datetime.now()
    current_pair: tuple[str, str] | None = None  # (reasoner_model, counter_model)

    for idx, row in matrix.iterrows():
        run_id = row["run_id"]
        r_model = row["reasoner_model"]
        c_model = row["counter_model"]
        seed = int(row["seed"])
        pair = (r_model, c_model)

        # Load both models for this pair if they changed.
        if pair != current_pair:
            models_to_load = list({r_model, c_model})
            # Unload models not needed in the new pair (never unload persistent models)
            if current_pair:
                for m in set(current_pair) - set(models_to_load):
                    if m not in persistent_models:
                        print(f"\n[vLLM] Unloading '{m}'…")
                        unload_model(m)
            load_models(models_to_load)
            current_pair = pair
            if not fixed_model:
                set_default_model(r_model)

        print(
            f"\n[{idx + 1}/{len(matrix)}] {run_id} | "
            f"R={r_model} C={c_model} | "
            f"plan=({row['planning_reasoner']},{row['planning_counter']}) | "
            f"caus={row['causality_enabled']} | rep={row['replicate']}"
        )

        _eval_model = evaluator_model or fixed_model
        payload = {
            "claim": row["claim_text"],
            "include_precedents": True,
            "max_statutes": 100,
            "max_precedents": 5,
            "settings": {
                "reasoner_model": r_model,
                "counter_model": c_model,
                **({"evaluator_model": _eval_model} if _eval_model else {}),
                "response_language": "en",
                "reasoner_temperature": 0.0,
                "counter_temperature": 0.3,
                "llm_max_tokens": 7168,
                "enable_planning_reasoner": bool(row["planning_reasoner"]),
                "enable_planning_counter": bool(row["planning_counter"]),
                "reasoner_enable_causality": bool(row.get("causality_enabled", True)),
                "counter_enable_causality": bool(row.get("causality_enabled", True)),
            },
        }

        t0 = time.time()
        try:
            response = _run_full_pipeline(payload)
            duration = time.time() - t0
            metrics = _extract_metrics(run_id, row.to_dict(), response, duration)
            metrics["status"] = "completed"
            metrics["error"] = ""
            metrics["started_at"] = datetime.now().isoformat()
            print(f"   ✅ {duration:.1f}s — verdict={metrics.get('aqa_verdict', '?')}")
        except Exception as exc:
            duration = time.time() - t0
            err_msg = str(exc)[:500]
            print(f"   ❌ FAILED {duration:.1f}s: {err_msg[:100]}")
            with open(log_path, "a") as lf:
                lf.write(f"{run_id}\t{err_msg}\n")
            metrics = {
                "run_id": run_id,
                "claim_id": row["claim_id"],
                "domain": row["domain"],
                "reasoner_model": r_model,
                "counter_model": c_model,
                "planning_reasoner": row["planning_reasoner"],
                "planning_counter": row["planning_counter"],
                "causality_enabled": bool(row.get("causality_enabled", True)),
                "replicate": row["replicate"],
                "seed": seed,
                "status": "failed",
                "error": err_msg,
                "duration_sec": round(duration, 2),
            }

        (out_dir / "runs" / f"run_{run_id}.json").write_text(
            json.dumps(
                {"row": row.to_dict(), "metrics": metrics}, indent=2, default=str
            )
        )
        all_metrics.append(metrics)
        pd.DataFrame(all_metrics).to_csv(out_dir / "metrics.csv", index=False)

    elapsed = (datetime.now() - started_at).total_seconds()
    completed = sum(1 for m in all_metrics if m.get("status") == "completed")
    failed = len(all_metrics) - completed

    print(f"\n{'='*60}")
    print(f"✅ DoE complete in {elapsed / 3600:.1f}h")
    print(f"   Completed: {completed}/{len(all_metrics)}")
    if failed:
        print(f"   Failed:    {failed}  (see {log_path})")
    print(f"   Results:   {out_dir}/metrics.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
