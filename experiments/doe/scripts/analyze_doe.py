#!/usr/bin/env python3
"""
Analyze DoE metrics (paired A/B) with lightweight non-parametric stats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sign_test_pvalue(diffs: np.ndarray) -> float | None:
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs != 0]
    n = diffs.size
    if n == 0:
        return None
    wins = int((diffs > 0).sum())
    losses = n - wins
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * p)


def mcnemar_exact_pvalue(paired: pd.DataFrame) -> float | None:
    needed = {"plausible_A", "plausible_B"}
    if paired.empty or not needed.issubset(set(paired.columns)):
        return None
    b = int(((paired["plausible_A"] == 0) & (paired["plausible_B"] == 1)).sum())
    c = int(((paired["plausible_A"] == 1) & (paired["plausible_B"] == 0)).sum())
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * p)


def bootstrap_mean_ci(
    diffs: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = diffs.size
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def summarize_metric(
    paired: pd.DataFrame,
    col_delta: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    if col_delta not in paired.columns:
        return {"metric": col_delta, "available": False}

    diffs = pd.to_numeric(paired[col_delta], errors="coerce").to_numpy(dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return {"metric": col_delta, "available": False}

    effective = diffs if higher_is_better else -diffs
    p_sign = sign_test_pvalue(effective)
    ci_lo, ci_hi = bootstrap_mean_ci(diffs)

    return {
        "metric": col_delta,
        "available": True,
        "n_pairs": int(diffs.size),
        "mean_delta_B_minus_A": float(np.mean(diffs)),
        "median_delta_B_minus_A": float(np.median(diffs)),
        "std_delta": float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0,
        "improvement_rate": float((effective > 0).mean()),
        "sign_test_pvalue": p_sign,
        "bootstrap_ci95_mean_delta": [ci_lo, ci_hi],
    }


def build_domain_breakdown(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    domains = sorted(x for x in metrics["domain"].dropna().unique())
    for domain in domains:
        ddf = metrics[metrics["domain"] == domain]
        for cond in ["A", "B"]:
            cdf = ddf[ddf["condition"] == cond]
            rows.append(
                {
                    "domain": domain,
                    "condition": cond,
                    "runs": int(cdf.shape[0]),
                    "aqa_net_final_mean": (
                        float(cdf["aqa_net_final"].mean())
                        if "aqa_net_final" in cdf
                        else None
                    ),
                    "reasoner_consistency_mean": (
                        float(cdf["reasoner_consistency_score"].mean())
                        if "reasoner_consistency_score" in cdf
                        else None
                    ),
                    "counter_consistency_mean": (
                        float(cdf["counter_consistency_score"].mean())
                        if "counter_consistency_score" in cdf
                        else None
                    ),
                    "counter_gate_abstain_rate": (
                        float(
                            pd.to_numeric(cdf["counter_gate_abstain"], errors="coerce")
                            .fillna(0)
                            .mean()
                        )
                        if "counter_gate_abstain" in cdf
                        else None
                    ),
                }
            )
    return rows


def build_stability_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if "replicate" not in metrics.columns:
        return {"available": False, "rows": rows}

    gcols = ["claim_id", "condition"]
    grouped = metrics.groupby(gcols, dropna=False)
    for (claim_id, condition), grp in grouped:
        net = pd.to_numeric(grp["aqa_net_final"], errors="coerce").dropna()
        verdicts = grp["aqa_verdict"].dropna().astype(str)
        if verdicts.empty:
            agreement = None
        else:
            agreement = float(verdicts.value_counts(normalize=True).max())
        rows.append(
            {
                "claim_id": claim_id,
                "condition": condition,
                "replicates": int(grp.shape[0]),
                "aqa_net_final_std": float(net.std(ddof=1)) if net.size > 1 else 0.0,
                "verdict_agreement": agreement,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"available": False, "rows": rows}
    out: dict[str, Any] = {
        "available": True,
        "rows": rows,
        "by_condition": {},
    }
    for cond in ["A", "B"]:
        cdf = df[df["condition"] == cond]
        if cdf.empty:
            continue
        out["by_condition"][cond] = {
            "mean_std_aqa_net_final": float(cdf["aqa_net_final_std"].mean()),
            "mean_verdict_agreement": (
                float(cdf["verdict_agreement"].dropna().mean())
                if cdf["verdict_agreement"].notna().any()
                else None
            ),
        }
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# DoE Analysis Summary")
    lines.append("")
    lines.append("## Global")
    global_stats = summary["global"]
    lines.append(f"- Runs: {global_stats['runs_total']}")
    lines.append(f"- Pairs: {global_stats['pairs_total']}")
    lines.append(f"- Plausible rate A: {global_stats['plausible_rate_A']:.3f}")
    lines.append(f"- Plausible rate B: {global_stats['plausible_rate_B']:.3f}")
    lines.append(f"- McNemar exact p-value: {global_stats['mcnemar_pvalue']}")
    lines.append("")
    lines.append("## Paired Metrics")
    for item in summary["paired_metrics"]:
        if not item.get("available"):
            lines.append(f"- {item['metric']}: n/a")
            continue
        lines.append(
            f"- {item['metric']}: mean_delta={item['mean_delta_B_minus_A']:.4f}, "
            f"improvement_rate={item['improvement_rate']:.3f}, "
            f"sign_p={item['sign_test_pvalue']}, "
            f"ci95={item['bootstrap_ci95_mean_delta']}"
        )
    lines.append("")
    lines.append("## Domain Breakdown")
    for row in summary["domain_breakdown"]:
        lines.append(
            f"- {row['domain']} {row['condition']}: runs={row['runs']}, "
            f"aqa_net_mean={row['aqa_net_final_mean']}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DoE metrics.")
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.csv path")
    parser.add_argument("--paired", type=Path, required=True, help="paired_deltas.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.metrics)
    paired = pd.read_csv(args.paired)

    metric_specs = [
        ("delta_aqa_net_final", True),
        ("delta_reasoner_consistency", True),
        ("delta_counter_consistency", True),
        ("delta_reasoner_repair_fail_rate", False),
        ("delta_counter_repair_fail_rate", False),
    ]
    paired_summary = [
        summarize_metric(paired, metric_name, higher_is_better=hib)
        for metric_name, hib in metric_specs
    ]

    if {"plausible_A", "plausible_B"}.issubset(
        set(paired.columns)
    ) and not paired.empty:
        plausible_rate_a = float((paired["plausible_A"] == 1).mean())
        plausible_rate_b = float((paired["plausible_B"] == 1).mean())
    else:
        plausible_rate_a = 0.0
        plausible_rate_b = 0.0

    summary = {
        "global": {
            "runs_total": int(metrics.shape[0]),
            "pairs_total": int(paired.shape[0]),
            "plausible_rate_A": plausible_rate_a,
            "plausible_rate_B": plausible_rate_b,
            "mcnemar_pvalue": mcnemar_exact_pvalue(paired),
        },
        "paired_metrics": paired_summary,
        "domain_breakdown": build_domain_breakdown(metrics),
        "stability": build_stability_summary(metrics),
    }

    json_path = args.out / "analysis_summary.json"
    md_path = args.out / "analysis_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    md_path.write_text(render_markdown(summary), "utf-8")

    print(f"[OK] {json_path}")
    print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
