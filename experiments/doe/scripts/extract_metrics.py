#!/usr/bin/env python3
"""
Extract DoE metrics from raw pipeline JSON responses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = PROJECT_ROOT / "experiments" / "doe" / "run_plan.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dig(data: Any, *keys: str, default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def count_repair_failed(report: dict[str, Any]) -> int:
    checks = report.get("citation_checks") or []
    if not isinstance(checks, list):
        return 0
    count = 0
    for c in checks:
        if not isinstance(c, dict):
            continue
        action = str(c.get("mismatch_action", "")).strip().lower()
        if action == "repair_failed":
            count += 1
    return count


def safe_div(n: float | int | None, d: float | int | None) -> float | None:
    if n is None or d is None or d == 0:
        return None
    return float(n) / float(d)


def extract_row_metrics(row: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    evaluation = dig(payload, "evaluation", default={}) or {}
    aqa = dig(evaluation, "aqa_report", default={}) or {}
    consistency = dig(evaluation, "consistency_report", default={}) or {}
    reasoner_report = dig(consistency, "reasoner", default={}) or {}
    counter_report = dig(consistency, "counter_reasoner", default={}) or {}

    gate = (
        dig(evaluation, "counter_reasoner_gate", default={})
        or dig(consistency, "counter_reasoner_gate", default={})
        or {}
    )
    counter = dig(payload, "counter_reasoner", default={}) or {}

    chain_scores = dig(aqa, "chain_scores", default={}) or {}
    chain_pro = dig(chain_scores, "pro", default={}) or {}
    chain_contra = dig(chain_scores, "contra", default={}) or {}
    net = dig(aqa, "net_plausibility", default={}) or {}

    reasoner_total = as_int(reasoner_report.get("total_citations"))
    counter_total = as_int(counter_report.get("total_citations"))
    reasoner_fail = count_repair_failed(reasoner_report)
    counter_fail = count_repair_failed(counter_report)

    return {
        "execution_index": as_int(row.get("execution_index")),
        "run_id": row.get("run_id"),
        "pair_id": row.get("pair_id"),
        "pair_order": row.get("pair_order"),
        "order_in_pair": as_int(row.get("order_in_pair")),
        "claim_id": row.get("claim_id"),
        "domain": row.get("domain"),
        "replicate": as_int(row.get("replicate")),
        "condition": row.get("condition"),
        "enable_causality": row.get("enable_causality"),
        "aqa_verdict": aqa.get("verdict"),
        "aqa_net_final": as_float(net.get("final")),
        "aqa_net_pro": as_float(net.get("pro")),
        "aqa_net_contra": as_float(net.get("contra")),
        "pro_cogency_avg": as_float(chain_pro.get("cogency_avg")),
        "pro_semantics_avg": as_float(chain_pro.get("semantics_avg")),
        "pro_norm_support_avg": as_float(chain_pro.get("norm_support_avg")),
        "contra_cogency_avg": as_float(chain_contra.get("cogency_avg")),
        "contra_semantics_avg": as_float(chain_contra.get("semantics_avg")),
        "contra_norm_support_avg": as_float(chain_contra.get("norm_support_avg")),
        "reasoner_consistency_score": as_float(
            reasoner_report.get("consistency_score")
        ),
        "counter_consistency_score": as_float(counter_report.get("consistency_score")),
        "reasoner_repaired_citations": as_int(
            reasoner_report.get("repaired_citations")
        ),
        "counter_repaired_citations": as_int(counter_report.get("repaired_citations")),
        "reasoner_dropped_citations": as_int(reasoner_report.get("dropped_citations")),
        "counter_dropped_citations": as_int(counter_report.get("dropped_citations")),
        "reasoner_repair_fail_count": reasoner_fail,
        "counter_repair_fail_count": counter_fail,
        "reasoner_repair_fail_rate": safe_div(reasoner_fail, reasoner_total),
        "counter_repair_fail_rate": safe_div(counter_fail, counter_total),
        "counter_gate_checked": gate.get("checked"),
        "counter_gate_label": gate.get("label"),
        "counter_gate_abstain": gate.get("abstain"),
        "counter_gate_reason": gate.get("reason"),
        "counter_planning_mode": counter.get("planning_mode"),
        "counter_reasoner_plan_hints_available": counter.get(
            "reasoner_plan_hints_available"
        ),
        "counter_selected_attacks_n": (
            len(counter.get("selected_attack_ids", []))
            if isinstance(counter.get("selected_attack_ids"), list)
            else None
        ),
    }


def build_paired_deltas(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_cols = ["pair_id", "claim_id", "domain", "replicate"]
    grouped = metrics_df.groupby(key_cols, dropna=False)

    for key, grp in grouped:
        if grp.shape[0] < 2:
            continue
        row_a = grp.loc[grp["condition"] == "A"]
        row_b = grp.loc[grp["condition"] == "B"]
        if row_a.empty or row_b.empty:
            continue
        a = row_a.iloc[0]
        b = row_b.iloc[0]

        paired = {
            "pair_id": key[0],
            "claim_id": key[1],
            "domain": key[2],
            "replicate": key[3],
            "aqa_net_final_A": a.get("aqa_net_final"),
            "aqa_net_final_B": b.get("aqa_net_final"),
            "delta_aqa_net_final": (
                b.get("aqa_net_final") - a.get("aqa_net_final")
                if pd.notna(a.get("aqa_net_final")) and pd.notna(b.get("aqa_net_final"))
                else None
            ),
            "reasoner_consistency_A": a.get("reasoner_consistency_score"),
            "reasoner_consistency_B": b.get("reasoner_consistency_score"),
            "delta_reasoner_consistency": (
                b.get("reasoner_consistency_score")
                - a.get("reasoner_consistency_score")
                if pd.notna(a.get("reasoner_consistency_score"))
                and pd.notna(b.get("reasoner_consistency_score"))
                else None
            ),
            "counter_consistency_A": a.get("counter_consistency_score"),
            "counter_consistency_B": b.get("counter_consistency_score"),
            "delta_counter_consistency": (
                b.get("counter_consistency_score") - a.get("counter_consistency_score")
                if pd.notna(a.get("counter_consistency_score"))
                and pd.notna(b.get("counter_consistency_score"))
                else None
            ),
            "reasoner_repair_fail_rate_A": a.get("reasoner_repair_fail_rate"),
            "reasoner_repair_fail_rate_B": b.get("reasoner_repair_fail_rate"),
            "delta_reasoner_repair_fail_rate": (
                b.get("reasoner_repair_fail_rate") - a.get("reasoner_repair_fail_rate")
                if pd.notna(a.get("reasoner_repair_fail_rate"))
                and pd.notna(b.get("reasoner_repair_fail_rate"))
                else None
            ),
            "counter_repair_fail_rate_A": a.get("counter_repair_fail_rate"),
            "counter_repair_fail_rate_B": b.get("counter_repair_fail_rate"),
            "delta_counter_repair_fail_rate": (
                b.get("counter_repair_fail_rate") - a.get("counter_repair_fail_rate")
                if pd.notna(a.get("counter_repair_fail_rate"))
                and pd.notna(b.get("counter_repair_fail_rate"))
                else None
            ),
            "aqa_verdict_A": a.get("aqa_verdict"),
            "aqa_verdict_B": b.get("aqa_verdict"),
            "plausible_A": int(str(a.get("aqa_verdict", "")).lower() == "plausible"),
            "plausible_B": int(str(b.get("aqa_verdict", "")).lower() == "plausible"),
            "counter_gate_abstain_A": a.get("counter_gate_abstain"),
            "counter_gate_abstain_B": b.get("counter_gate_abstain"),
        }
        rows.append(paired)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DoE metrics from raw runs.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="run_plan.csv")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder generated by run_doe.py (contains raw/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for metrics CSV/Parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plan_rows = read_csv(args.plan)
    raw_dir = args.run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    extracted: list[dict[str, Any]] = []
    missing_raw = 0
    for row in plan_rows:
        run_id = row["run_id"]
        raw_file = raw_dir / f"{run_id}.json"
        if not raw_file.exists():
            missing_raw += 1
            extracted.append(
                {
                    "execution_index": as_int(row.get("execution_index")),
                    "run_id": run_id,
                    "pair_id": row.get("pair_id"),
                    "pair_order": row.get("pair_order"),
                    "order_in_pair": as_int(row.get("order_in_pair")),
                    "claim_id": row.get("claim_id"),
                    "domain": row.get("domain"),
                    "replicate": as_int(row.get("replicate")),
                    "condition": row.get("condition"),
                    "enable_causality": row.get("enable_causality"),
                    "raw_missing": True,
                }
            )
            continue

        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        row_metrics = extract_row_metrics(row, payload)
        row_metrics["raw_missing"] = False
        extracted.append(row_metrics)

    metrics_df = pd.DataFrame(extracted)
    metrics_csv = args.out / "metrics.csv"
    metrics_parquet = args.out / "metrics.parquet"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
    metrics_df.to_parquet(metrics_parquet, index=False)

    paired_df = build_paired_deltas(metrics_df[metrics_df["raw_missing"].eq(False)])
    paired_csv = args.out / "paired_deltas.csv"
    paired_df.to_csv(paired_csv, index=False, encoding="utf-8")

    summary = {
        "plan_rows": len(plan_rows),
        "metrics_rows": int(metrics_df.shape[0]),
        "paired_rows": int(paired_df.shape[0]),
        "missing_raw_rows": int(missing_raw),
        "metrics_csv": str(metrics_csv),
        "paired_csv": str(paired_csv),
    }
    (args.out / "extract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] metrics: {metrics_csv}")
    print(f"[OK] paired:  {paired_csv}")
    print(f"[INFO] missing raw: {missing_raw}")


if __name__ == "__main__":
    main()
