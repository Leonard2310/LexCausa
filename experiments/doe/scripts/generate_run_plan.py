#!/usr/bin/env python3
"""
Generate a paired randomized blocked run plan for LexCausa DoE.

Conditions:
- A: enable_causality=false
- B: enable_causality=true
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLAIMS_FILE = PROJECT_ROOT / "claims.md"
DEFAULT_OUT_CSV = PROJECT_ROOT / "experiments" / "doe" / "run_plan.csv"


@dataclass
class ClaimCase:
    claim_id: str
    title: str
    text: str
    domain: str
    covered: bool


def _normalize_domain_from_heading(heading: str) -> str:
    h = heading.upper()
    if "CIVILI" in h:
        return "CIVILE"
    if "PENALI" in h:
        return "PENALE"
    if "AMMINISTRATIVI" in h:
        return "AMMINISTRATIVO"
    if "MIXED" in h:
        return "MIXED"
    if "NON COPERTI" in h:
        return "NON_COPERTO"
    return "UNKNOWN"


def parse_claims_md(claims_path: Path) -> list[ClaimCase]:
    lines = claims_path.read_text(encoding="utf-8").splitlines()

    block_domain = "UNKNOWN"
    block_covered = False

    claims: list[ClaimCase] = []
    current_id = ""
    current_title = ""
    current_text_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_id, current_title, current_text_lines
        if not current_id:
            return
        text = re.sub(r"\s+", " ", " ".join(current_text_lines)).strip()
        if text:
            claims.append(
                ClaimCase(
                    claim_id=current_id,
                    title=current_title,
                    text=text,
                    domain=block_domain,
                    covered=block_covered,
                )
            )
        current_id = ""
        current_title = ""
        current_text_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("## "):
            flush_current()
            block_domain = _normalize_domain_from_heading(line)
            block_covered = "(COPERTI)" in line.upper()
            continue

        match_claim = re.match(r"^###\s+([A-Z0-9]+)\s*-\s*(.+)$", line)
        if match_claim:
            flush_current()
            current_id = match_claim.group(1).strip()
            current_title = match_claim.group(2).strip()
            continue

        if line.startswith("- Gap normativo:"):
            continue
        if line.startswith("Testo completo:"):
            line = line.replace("Testo completo:", "", 1).strip()

        if current_id:
            current_text_lines.append(line)

    flush_current()
    return claims


def build_run_plan(
    claims: list[ClaimCase],
    *,
    domains: list[str],
    include_non_covered: bool,
    include_mixed: bool,
    replicates: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)

    eligible: list[ClaimCase] = []
    allowed_domains = {d.strip().upper() for d in domains if d.strip()}
    for c in claims:
        if c.domain == "MIXED" and not include_mixed:
            continue
        if c.domain == "NON_COPERTO" and not include_non_covered:
            continue
        if c.domain not in allowed_domains:
            continue
        if not include_non_covered and not c.covered:
            continue
        eligible.append(c)

    if not eligible:
        raise ValueError("No eligible claims found. Check domains/flags.")

    grouped_pairs: dict[str, list[tuple[ClaimCase, int, str]]] = {}
    for claim in eligible:
        grouped_pairs.setdefault(claim.domain, [])
        for replicate in range(1, replicates + 1):
            pair_order = "AB" if rng.random() < 0.5 else "BA"
            grouped_pairs[claim.domain].append((claim, replicate, pair_order))

    for domain in grouped_pairs:
        rng.shuffle(grouped_pairs[domain])

    rows: list[dict[str, str]] = []
    execution_index = 1
    for domain in domains:
        d = domain.upper().strip()
        for claim, replicate, pair_order in grouped_pairs.get(d, []):
            pair_id = f"{claim.claim_id}_R{replicate}"
            if pair_order == "AB":
                cond_sequence = [("A", "false"), ("B", "true")]
            else:
                cond_sequence = [("B", "true"), ("A", "false")]

            for order_in_pair, (condition, enable_causality) in enumerate(
                cond_sequence, start=1
            ):
                run_id = (
                    f"{claim.claim_id}_R{replicate}_{condition}_"
                    f"{execution_index:04d}"
                )
                rows.append(
                    {
                        "execution_index": str(execution_index),
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "pair_order": pair_order,
                        "order_in_pair": str(order_in_pair),
                        "claim_id": claim.claim_id,
                        "claim_title": claim.title,
                        "claim_text": claim.text,
                        "domain": claim.domain,
                        "replicate": str(replicate),
                        "condition": condition,
                        "enable_causality": enable_causality,
                    }
                )
                execution_index += 1

    return rows


def write_plan(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "execution_index",
        "run_id",
        "pair_id",
        "pair_order",
        "order_in_pair",
        "claim_id",
        "claim_title",
        "claim_text",
        "domain",
        "replicate",
        "condition",
        "enable_causality",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], summary_path: Path) -> None:
    domains_count: dict[str, int] = {}
    claims_count: dict[str, set[str]] = {}
    for row in rows:
        domain = row["domain"]
        domains_count[domain] = domains_count.get(domain, 0) + 1
        claims_count.setdefault(domain, set()).add(row["claim_id"])

    summary = {
        "total_runs": len(rows),
        "total_pairs": len({r["pair_id"] for r in rows}),
        "domains": {
            d: {
                "runs": domains_count.get(d, 0),
                "unique_claims": len(claims_count.get(d, set())),
            }
            for d in sorted(domains_count.keys())
        },
        "conditions": {
            "A": sum(1 for r in rows if r["condition"] == "A"),
            "B": sum(1 for r in rows if r["condition"] == "B"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DoE run plan CSV.")
    parser.add_argument(
        "--claims-file",
        type=Path,
        default=DEFAULT_CLAIMS_FILE,
        help="Path to claims.md",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output CSV path for run plan",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="Replicates per claim (>=1)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="CIVILE,PENALE,AMMINISTRATIVO",
        help="Comma-separated domain order/selection",
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Include MIXED claims as additional domain",
    )
    parser.add_argument(
        "--include-non-covered",
        action="store_true",
        help="Include NON_COPERTO claims (NC*)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for AB/BA randomization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    claims = parse_claims_md(args.claims_file)
    domains = [d.strip().upper() for d in args.domains.split(",") if d.strip()]
    rows = build_run_plan(
        claims,
        domains=domains,
        include_non_covered=args.include_non_covered,
        include_mixed=args.include_mixed,
        replicates=args.replicates,
        seed=args.seed,
    )

    write_plan(rows, args.out)
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    write_summary(rows, summary_path)

    print(f"[OK] Run plan written: {args.out}")
    print(f"[OK] Summary written:  {summary_path}")
    print(f"[INFO] Total runs: {len(rows)}")


if __name__ == "__main__":
    main()
