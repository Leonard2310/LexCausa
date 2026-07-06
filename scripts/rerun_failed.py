#!/usr/bin/env python3
"""Re-run only the failed rows of a completed multi-DoE shard.

A shard writes ``metrics.csv`` once, at the end, with ``status="failed"`` for
the runs that hit a transient provider error (e.g. a 429/500 rate-limit). This
helper reads that ``metrics.csv``, finds the failed ``run_id``s, replays exactly
those configurations against a healthy backend, and rewrites ``metrics.csv`` in
place with the repaired rows. Everything else is left untouched.

Usage (one shard at a time, pointing at that shard's backend):

    poetry run python scripts/rerun_failed.py \
        --run-dir experiments/multi_doe/runs/or_sc240_s1 \
        --api-url http://localhost:8001

Run with ``--dry-run`` first to see which runs would be replayed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multi_doe import MultiDoE  # noqa: E402


def _load_failed(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metrics_path = run_dir / "metrics.csv"
    matrix_path = run_dir / "run_matrix.csv"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.csv in {run_dir} (shard not finished yet?)")
    if not matrix_path.exists():
        raise SystemExit(f"No run_matrix.csv in {run_dir}")

    metrics = pd.read_csv(metrics_path)
    matrix = pd.read_csv(matrix_path)
    if "status" not in metrics.columns:
        raise SystemExit("metrics.csv has no 'status' column")

    failed_ids = metrics.loc[
        metrics["status"].astype(str).str.lower() == "failed", "run_id"
    ].tolist()
    return metrics, matrix, failed_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="A single shard directory")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--max-statutes", type=int, default=100)
    ap.add_argument("--max-precedents", type=int, default=5)
    ap.add_argument("--min-kept", type=int, default=None)
    ap.add_argument(
        "--claims-file",
        default="claims_doe12.md",
        help="Claims markdown of the campaign, used to recover claim_text",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics, matrix, failed_ids = _load_failed(run_dir)

    if not failed_ids:
        print(f"✅ No failed runs in {run_dir}. Nothing to do.")
        return

    print(f"Found {len(failed_ids)} failed run(s) in {run_dir.name}:")
    rows = matrix[matrix["run_id"].isin(failed_ids)]
    for _, r in rows.iterrows():
        print(
            f"  {r['run_id']} | {r['claim_id']} | R:{r['reasoner_model']} "
            f"C:{r['counter_model']} | paradigm:{r.get('paradigm','?')} "
            f"| rep:{r['replicate']}"
        )
    if args.dry_run:
        print("\n(dry-run) Nothing executed.")
        return

    # A minimal runner instance: we only use _execute_pipeline_run,
    # _extract_metrics_from_response and _persist_raw_response, so the model /
    # claim lists are irrelevant (no matrix is generated here).
    runner = MultiDoE(
        claims_file=args.claims_file,
        reasoner_models=["gpt_oss_120b"],
        counter_models=["gpt_oss_120b"],
        domains=["CIVILE"],
        planning_ablations=[(False, False, True)],
        replicates=1,
        output_dir=str(run_dir),
        api_url=args.api_url,
        max_statutes=args.max_statutes,
        max_precedents=args.max_precedents,
        min_kept=args.min_kept,
    )

    # The run_matrix is overwritten with results at the end of a shard, and the
    # failed rows there lose single_call/paradigm. Backfill them from the shard's
    # completed rows (these flags are constant per shard), so a single-call run is
    # not accidentally replayed as a plain step-wise one.
    comp = metrics[metrics["status"].astype(str).str.lower() == "completed"]

    def _fill(col: str):
        if col in comp.columns:
            vals = comp[col].dropna()
            if not vals.empty:
                return vals.mode().iloc[0]
        return None

    backfill = {
        c: _fill(c) for c in ("single_call_reasoner", "single_call_counter", "paradigm")
    }

    # The overwritten run_matrix also drops claim_text (and can drop domain) on
    # failed rows; recover them from the claims file by claim_id.
    claim_text = {c["id"]: c["text"] for c in runner.claims}
    claim_dom = {c["id"]: c["domain"] for c in runner.claims}

    repaired: dict[str, dict] = {}
    for _, row in rows.iterrows():
        rd = row.to_dict()
        for c, v in backfill.items():
            if v is not None and (c not in rd or pd.isna(rd.get(c))):
                rd[c] = v
        if not rd.get("claim_text") or pd.isna(rd.get("claim_text")):
            rd["claim_text"] = claim_text.get(rd["claim_id"])
        if pd.isna(rd.get("domain")):
            rd["domain"] = claim_dom.get(rd["claim_id"], rd.get("domain"))
        rid = rd["run_id"]
        print(f"\n▶ Replaying {rid} ({rd['claim_id']}, rep {rd['replicate']})")
        try:
            started = datetime.now()
            response = runner._execute_pipeline_run(rd)
            duration = (datetime.now() - started).total_seconds()
            m = runner._extract_metrics_from_response(response, rd)
            m["run_id"] = rid
            m["status"] = "completed"
            m["duration_sec"] = duration
            m["started_at"] = started.isoformat()
            m["completed_at"] = datetime.now().isoformat()
            runner._persist_raw_response(rid, response)
            repaired[rid] = m
            print(f"   ✅ Completed in {duration:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"   ❌ Still failing: {str(e)[:150]}")

    if not repaired:
        print("\nNo runs repaired. metrics.csv left unchanged.")
        return

    # Splice repaired rows back into metrics.csv (match columns, keep order).
    repaired_df = pd.DataFrame(list(repaired.values()))
    keep = metrics[~metrics["run_id"].isin(repaired.keys())]
    merged = pd.concat([keep, repaired_df], ignore_index=True)
    # Preserve original run_id ordering from the matrix where possible.
    order = {rid: i for i, rid in enumerate(matrix["run_id"].tolist())}
    merged["_ord"] = merged["run_id"].map(order).fillna(1e9)
    merged = merged.sort_values("_ord").drop(columns="_ord")

    backup = run_dir / "metrics.pre_rerun.csv"
    metrics.to_csv(backup, index=False)
    merged.to_csv(run_dir / "metrics.csv", index=False)
    print(
        f"\n📊 Repaired {len(repaired)}/{len(failed_ids)} run(s). "
        f"Updated metrics.csv (backup: {backup.name})."
    )


if __name__ == "__main__":
    main()
