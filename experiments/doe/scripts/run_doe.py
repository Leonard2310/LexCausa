#!/usr/bin/env python3
"""
Run LexCausa DoE plan against /api/pipeline and persist raw responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = PROJECT_ROOT / "experiments" / "doe" / "run_plan.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "doe" / "doe_settings.json"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "experiments" / "doe" / "runs"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_run_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310
            status = int(resp.getcode())
            text = resp.read().decode("utf-8", errors="replace")
            return status, text
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), text


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"raw_response": data}
    except json.JSONDecodeError:
        return {"raw_response": text}


def ensure_status_file(status_csv: Path) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    if status_csv.exists():
        return
    with status_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "execution_index",
                "run_id",
                "claim_id",
                "domain",
                "replicate",
                "condition",
                "enable_causality",
                "status",
                "http_status",
                "attempts",
                "duration_sec",
                "raw_file",
                "error_message",
            ],
        )
        writer.writeheader()


def append_status(status_csv: Path, row: dict[str, Any]) -> None:
    with status_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "execution_index",
                "run_id",
                "claim_id",
                "domain",
                "replicate",
                "condition",
                "enable_causality",
                "status",
                "http_status",
                "attempts",
                "duration_sec",
                "raw_file",
                "error_message",
            ],
        )
        writer.writerow(row)


def build_payload(
    plan_row: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(cfg.get("settings", {}))
    condition_enabled = as_bool(plan_row["enable_causality"])
    # Legacy global switch retained for backward compatibility.
    settings["enable_causality"] = condition_enabled
    # DoE isolation: by default toggle causality on Counter only.
    settings["reasoner_enable_causality"] = as_bool(
        settings.get("reasoner_enable_causality", True)
    )
    settings["counter_enable_causality"] = condition_enabled
    # Counter inputs are opt-in and default disabled (can be overridden in config).
    pass_attacks_cfg = as_bool(settings.get("counter_pass_taxonomy_attacks", False))
    pass_norms_cfg = as_bool(settings.get("counter_pass_norms", False))
    settings["counter_pass_taxonomy_attacks"] = condition_enabled and pass_attacks_cfg
    settings["counter_pass_norms"] = condition_enabled and pass_norms_cfg

    return {
        "claim": plan_row["claim_text"],
        "include_precedents": bool(cfg.get("include_precedents", True)),
        "max_statutes": int(cfg.get("max_statutes", 100)),
        "max_precedents": int(cfg.get("max_precedents", 5)),
        "settings": settings,
    }


def run_plan(
    plan_rows: list[dict[str, str]],
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    resume: bool,
    only_condition: str | None,
    start_from: int,
    max_runs: int | None,
) -> None:
    raw_dir = run_dir / "raw"
    req_dir = run_dir / "requests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)

    status_csv = run_dir / "run_status.csv"
    ensure_status_file(status_csv)

    endpoint = str(cfg.get("endpoint", "http://127.0.0.1:8000/api/pipeline")).strip()
    timeout_sec = int(cfg.get("timeout_sec", 1800))
    max_retries = int(cfg.get("max_retries", 3))
    retry_backoff_sec = int(cfg.get("retry_backoff_sec", 20))

    executed = 0
    for row in sorted(plan_rows, key=lambda r: int(r["execution_index"])):
        exec_idx = int(row["execution_index"])
        if exec_idx < start_from:
            continue
        if only_condition and row["condition"].upper() != only_condition.upper():
            continue
        if max_runs is not None and executed >= max_runs:
            break

        run_id = row["run_id"]
        raw_file = raw_dir / f"{run_id}.json"
        req_file = req_dir / f"{run_id}.json"
        if resume and raw_file.exists():
            print(f"[SKIP] {run_id} raw exists")
            continue

        payload = build_payload(row, cfg)
        req_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        start = time.time()
        status = "failed"
        http_status = 0
        err_msg = ""
        response_data: dict[str, Any] = {}

        for attempt in range(1, max_retries + 1):
            http_status, resp_text = post_json(endpoint, payload, timeout_sec)
            response_data = parse_response_json(resp_text)

            if 200 <= http_status < 300:
                status = "ok"
                err_msg = ""
                break

            err_msg = str(response_data.get("error", resp_text))[:1000]
            print(
                f"[WARN] run={run_id} attempt={attempt}/{max_retries} "
                f"http={http_status} err={err_msg[:120]}"
            )
            if attempt < max_retries:
                time.sleep(retry_backoff_sec * attempt)

        duration = round(time.time() - start, 3)

        raw_file.write_text(
            json.dumps(response_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        append_status(
            status_csv,
            {
                "execution_index": exec_idx,
                "run_id": run_id,
                "claim_id": row["claim_id"],
                "domain": row["domain"],
                "replicate": row["replicate"],
                "condition": row["condition"],
                "enable_causality": row["enable_causality"],
                "status": status,
                "http_status": http_status,
                "attempts": max_retries if status != "ok" else attempt,
                "duration_sec": duration,
                "raw_file": str(raw_file.relative_to(run_dir)),
                "error_message": err_msg,
            },
        )
        executed += 1
        print(
            f"[{status.upper()}] {run_id} http={http_status} "
            f"duration={duration}s raw={raw_file.name}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute DoE run plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="run_plan.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="DoE config JSON (copy from template)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root folder for run artifacts",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional explicit run folder id; default timestamp",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from execution_index",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Stop after N executed runs",
    )
    parser.add_argument(
        "--only-condition",
        type=str,
        default="",
        help="Optional filter: A or B",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip runs with existing raw files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_rows = read_run_plan(args.plan)
    cfg = load_json(args.config)

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Plan rows: {len(plan_rows)}")
    print(f"[INFO] Run dir:   {run_dir}")
    print(f"[INFO] Endpoint:  {cfg.get('endpoint')}")

    run_plan(
        plan_rows=plan_rows,
        cfg=cfg,
        run_dir=run_dir,
        resume=not args.no_resume,
        only_condition=(args.only_condition.strip().upper() or None),
        start_from=args.start_from,
        max_runs=args.max_runs,
    )
    print("[OK] DoE run completed.")


if __name__ == "__main__":
    main()
