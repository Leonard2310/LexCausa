#!/usr/bin/env python3
"""
Capture /api/chat retrieval outputs for all covered claims and persist them.

Can also warm the backend claim-context memory (SQLite pre-retrieval cache)
because /api/chat reuses the same `prepare_claim_context(...)` path as the
Reasoner/Counter pipeline.

For each claim in claims.md (covered sections), this script calls the local API
endpoint /api/chat with precedents enabled and stores:
- filtered statutes returned by the endpoint
- filtered precedents returned by the endpoint
- classification metadata
- request metadata and status

Outputs:
- logs/api_chat_memory/<timestamp>_<claim_id>_<slug>.json (one per claim)
- logs/api_chat_memory/<timestamp>_manifest.json (aggregate run report)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_MD_PATH = PROJECT_ROOT / "claims.md"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "api_chat_memory"


@dataclass
class ClaimEntry:
    claim_id: str
    section: str
    domain: str
    title: str
    text: str


def _slugify_filename(text: str, max_len: int = 60) -> str:
    clean = (text or "").strip().replace("\n", " ")
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in clean)
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe[:max_len] or "claim").strip("_")


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
    pattern: re.Pattern[str],
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
        claims.append(
            ClaimEntry(
                claim_id=str(match.group(1)).strip().upper(),
                title=str(match.group(2)).strip(),
                text=body,
                section=section,
                domain=domain,
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

    claims = (
        _parse_claims_from_block(
            civile_block, claim_pattern, "CLAIM CIVILI (COPERTI)", "civile"
        )
        + _parse_claims_from_block(
            penale_block, claim_pattern, "CLAIM PENALI (COPERTI)", "penale"
        )
        + _parse_claims_from_block(
            mixed_block, claim_pattern, "CLAIM MIXED (COPERTI)", "misto"
        )
        + _parse_claims_from_block(
            admin_block,
            claim_pattern,
            "CLAIM AMMINISTRATIVI (COPERTI)",
            "amministrativo",
        )
    )

    non_covered_count = len(claim_pattern.findall(non_covered_block))
    return claims, non_covered_count


def _healthcheck(base_url: str, timeout_s: float = 10.0) -> None:
    url = f"{base_url.rstrip('/')}/health"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()


def _call_api_chat(
    base_url: str,
    claim: str,
    top_k: int,
    max_precedents: int,
    timeout_s: float | None,
    settings_payload: dict[str, Any] | None = None,
    claim_context_memory_enabled: bool = False,
    claim_context_memory_overwrite: bool = False,
) -> requests.Response:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "message": claim,
        "top_k": top_k,
        "include_precedents": True,
        "max_precedents": max_precedents,
    }
    if settings_payload:
        payload["settings"] = settings_payload
    if claim_context_memory_enabled:
        payload["claim_context_memory_enabled"] = True
    if claim_context_memory_overwrite:
        payload["claim_context_memory_overwrite"] = True
    return requests.post(url, json=payload, timeout=timeout_s)


def _parse_timeout_arg(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "no", "null", "inf", "infinite"}:
        return None
    parsed = float(raw)
    if parsed <= 0:
        return None
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-precedents", type=int, default=5)
    parser.add_argument(
        "--timeout",
        type=_parse_timeout_arg,
        default=None,
        help="HTTP timeout seconds. Use 'none' (default) to disable timeout.",
    )
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument(
        "--domains",
        nargs="*",
        default=[],
        help="Optional subset: penale civile amministrativo misto",
    )
    parser.add_argument(
        "--search-query-terms-mode",
        default="llm",
        choices=["llm"],
        help="Forwarded to /api/chat settings. Kept explicit for reproducibility.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip claims that already have a saved file in logs/api_chat_memory/*_<CLAIM_ID>_*.json",
    )
    parser.add_argument(
        "--claim-context-memory",
        action="store_true",
        help="Enable backend SQLite claim-context memory while calling /api/chat (warm cache).",
    )
    parser.add_argument(
        "--overwrite-claim-context-memory",
        action="store_true",
        help="Force overwrite of existing claim-context memory entries (implies --claim-context-memory).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Warm backend claim-context memory but skip writing per-claim JSON capture files.",
    )
    args = parser.parse_args()

    if args.overwrite_claim_context_memory:
        args.claim_context_memory = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    claims, non_covered_count = parse_claims_md(CLAIMS_MD_PATH)
    domain_filter = {d.strip().lower() for d in args.domains if d.strip()}
    if domain_filter:
        claims = [c for c in claims if c.domain in domain_filter]

    if not claims:
        print("No claims selected.")
        return 1

    try:
        _healthcheck(args.base_url)
    except Exception as exc:
        print(f"Healthcheck failed on {args.base_url}: {exc}")
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = OUTPUT_DIR / f"{ts}_manifest.json"

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "claims_path": str(CLAIMS_MD_PATH),
        "non_covered_excluded": non_covered_count,
        "request": {
            "top_k": int(args.top_k),
            "include_precedents": True,
            "max_precedents": int(args.max_precedents),
            "settings": {"search_query_terms_mode": args.search_query_terms_mode},
            "timeout": None if args.timeout is None else float(args.timeout),
            "delay": float(args.delay),
            "claim_context_memory_enabled": bool(args.claim_context_memory),
            "claim_context_memory_overwrite": bool(args.overwrite_claim_context_memory),
            "cache_only": bool(args.cache_only),
        },
        "counts": {"selected_claims": len(claims)},
        "results": [],
    }

    ok = 0
    failed = 0
    existing_claim_ids: set[str] = set()
    if args.skip_existing:
        for p in OUTPUT_DIR.glob("*.json"):
            m = re.search(r"_([A-Z]\d+)_", p.name)
            if m and not p.name.endswith("_manifest.json"):
                existing_claim_ids.add(m.group(1).upper())
        manifest["resume"] = {
            "skip_existing": True,
            "existing_claim_ids_detected": sorted(existing_claim_ids),
        }

    for idx, entry in enumerate(claims, start=1):
        if entry.claim_id.upper() in existing_claim_ids:
            skipped_record = {
                "claim_id": entry.claim_id,
                "domain": entry.domain,
                "title": entry.title,
                "status": "skipped_existing",
            }
            manifest["results"].append(skipped_record)
            manifest["counts"]["skipped_existing"] = (
                manifest["counts"].get("skipped_existing", 0) + 1
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{idx}/{len(claims)}] {entry.claim_id} ({entry.domain}) - skipped existing",
                flush=True,
            )
            continue

        print(
            f"[{idx}/{len(claims)}] {entry.claim_id} ({entry.domain}) - {entry.title}",
            flush=True,
        )
        started = time.time()
        record: dict[str, Any] = {
            "claim_id": entry.claim_id,
            "domain": entry.domain,
            "title": entry.title,
            "claim": entry.text,
            "status": "error",
        }
        try:
            resp = _call_api_chat(
                base_url=args.base_url,
                claim=entry.text,
                top_k=int(args.top_k),
                max_precedents=int(args.max_precedents),
                timeout_s=None if args.timeout is None else float(args.timeout),
                settings_payload={
                    "search_query_terms_mode": args.search_query_terms_mode
                },
                claim_context_memory_enabled=bool(args.claim_context_memory),
                claim_context_memory_overwrite=bool(
                    args.overwrite_claim_context_memory
                ),
            )
            elapsed = round(time.time() - started, 3)
            record["elapsed_s"] = elapsed
            record["http_status"] = resp.status_code

            if resp.status_code != 200:
                text = resp.text
                record["error"] = text[:4000]
                failed += 1
            else:
                payload = resp.json()
                statutes = payload.get("articles") if isinstance(payload, dict) else []
                precedents = (
                    payload.get("precedents") if isinstance(payload, dict) else []
                )
                if not isinstance(statutes, list):
                    statutes = []
                if not isinstance(precedents, list):
                    precedents = []

                record.update(
                    {
                        "status": "ok",
                        "articles_count": len(statutes),
                        "precedents_count": len(precedents),
                    }
                )
                if not args.cache_only:
                    claim_file = OUTPUT_DIR / (
                        f"{ts}_{entry.claim_id}_{_slugify_filename(entry.title, max_len=40)}.json"
                    )
                    claim_payload = {
                        "captured_at": datetime.now().isoformat(timespec="seconds"),
                        "claim_meta": asdict(entry),
                        "request": manifest["request"],
                        "http_status": resp.status_code,
                        "classification": payload.get("classification"),
                        "articles": statutes,
                        "precedents": precedents,
                        "counts": {
                            "articles": len(statutes),
                            "precedents": len(precedents),
                        },
                    }
                    claim_file.write_text(
                        json.dumps(claim_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    record["file"] = {
                        "absolute_path": str(claim_file),
                        "relative_path": str(claim_file.relative_to(PROJECT_ROOT)),
                    }
                ok += 1

        except Exception as exc:
            record["elapsed_s"] = round(time.time() - started, 3)
            record["error"] = str(exc)
            failed += 1

        manifest["results"].append(record)
        manifest["counts"]["ok"] = ok
        manifest["counts"]["failed"] = failed

        # Incremental checkpoint after each claim.
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.delay > 0 and idx < len(claims):
            time.sleep(float(args.delay))

    print(f"\nSaved manifest: {manifest_path}")
    print(
        "OK={} | FAILED={} | SKIPPED={}".format(
            ok, failed, manifest["counts"].get("skipped_existing", 0)
        )
    )
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
