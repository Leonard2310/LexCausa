#!/usr/bin/env python
"""
Advanced Design of Experiments (DoE) Framework for LexCausa.

Supports multi-dimensional ablation studies:
- RQ1: Model Efficacy (comparing reasoning vs non-reasoning models)
- RQ2: Citation Faithfulness (accuracy of cited statutes)
- RQ3: Planning Ablation (impact of planning on reasoning quality)
- Auxiliary: Token Cost Analysis

Usage:
    python scripts/run_multi_doe.py \\
        --claims-file claims.md \\
        --models gpt_oss_120b,groq_llama_scout_17b \\
        --domains CIVILE,PENALE,AMMINISTRATIVO \\
        --planning-ablations on,off \\
        --replicates 10 \\
        --docker \\
        --out experiments/multi_doe/runs/$(date +%Y%m%d_%H%M%S) \\
        --seed 42
"""

import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# API endpoint
API_BASE = "http://localhost:8000/api"


class MultiDoE:
    """Multi-DoE orchestration for LexCausa."""

    def __init__(
        self,
        claims_file: str,
        models: list[str],
        domains: list[str],
        planning_ablations: list[tuple[bool, bool]],
        replicates: int,
        output_dir: str,
        use_docker: bool = False,
        seed: Optional[int] = None,
        causality_ablations: Optional[list[bool]] = None,
        causality_models: Optional[list[str]] = None,
    ):
        """Initialize DoE framework.

        causality_ablations: which causality conditions to test (e.g. [True, False]).
            Defaults to [True] (always on — no ablation).
        causality_models: models for which causality ablation is applied.
            Defaults to ["gpt_oss_120b"]. Other models always run with causality=True.
        """
        self.claims_file = Path(claims_file)
        self.models = models
        self.domains = domains
        self.planning_ablations = planning_ablations  # [(True, True), (False, False)]
        self.causality_ablations = (
            causality_ablations if causality_ablations is not None else [True]
        )
        self.causality_models = set(causality_models or ["gpt_oss_120b"])
        self.replicates = replicates
        self.output_dir = Path(output_dir)
        self.use_docker = use_docker
        self.seed = seed or 42

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "runs").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Load claims
        self.claims = self._load_claims()
        print(f"✅ Loaded {len(self.claims)} claims from {claims_file}")

    def _load_claims(self) -> list[dict]:
        """Load claims from markdown file.

        Expected format:
            ## CLAIM CIVILI (COPERTI)   <- domain section header
            ### C1 - Title              <- individual claim
            Claim body text...

            ### C2 - Title
            ...
        """
        claims = []
        with open(self.claims_file, "r") as f:
            content = f.read()

        # Map section header keywords → canonical domain names
        _DOMAIN_MAP = {
            "CIVILI": "CIVILE",
            "PENALI": "PENALE",
            "AMMINISTRATIVI": "AMMINISTRATIVO",
            "MIXED": "MISTO",
            "MISTO": "MISTO",
        }

        current_domain = "MISTO"
        current_claim_id: str | None = None
        current_claim_title: str | None = None
        current_claim_lines: list[str] = []

        def _flush():
            if current_claim_id is None:
                return
            text = "\n".join(current_claim_lines).strip()
            claims.append(
                {
                    "id": current_claim_id,
                    "domain": current_domain,
                    "title": current_claim_title,
                    "text": text[:300],  # Truncate for practical runs
                }
            )

        for line in content.splitlines():
            if line.startswith("## "):
                # Domain section header — e.g. "## CLAIM CIVILI (COPERTI)"
                header = line[3:].upper()
                for keyword, domain in _DOMAIN_MAP.items():
                    if keyword in header:
                        current_domain = domain
                        break
            elif line.startswith("### "):
                # Individual claim — flush previous and start new
                _flush()
                current_claim_lines = []
                current_claim_title = line[4:].strip()
                # Derive a short ID from the first token (e.g. "C1", "P3", "A2")
                first_token = (
                    current_claim_title.split()[0] if current_claim_title else ""
                )
                current_claim_id = (
                    first_token if first_token else f"claim_{len(claims):03d}"
                )
            elif current_claim_id is not None:
                current_claim_lines.append(line)

        _flush()  # flush last claim
        return claims

    def generate_run_matrix(self) -> pd.DataFrame:
        """Generate Cartesian product of all experimental conditions."""
        matrix = []

        for claim in self.claims:
            # Filter claims by domain if specified
            if self.domains and claim["domain"] not in self.domains:
                continue

            for model in self.models:
                # Causality ablation only for designated models (e.g. OSS120b)
                causality_conditions = (
                    self.causality_ablations
                    if model in self.causality_models
                    else [True]
                )

                for plan_r, plan_c in self.planning_ablations:
                    for causality_on in causality_conditions:
                        for rep in range(self.replicates):
                            matrix.append(
                                {
                                    "run_id": str(uuid.uuid4().hex[:12]),
                                    "claim_id": claim["id"],
                                    "claim_text": claim["text"],
                                    "domain": claim["domain"],
                                    "model": model,
                                    "planning_reasoner": plan_r,
                                    "planning_counter": plan_c,
                                    "causality_enabled": causality_on,
                                    "replicate": rep,
                                    "status": "pending",
                                    "error": "",
                                    "started_at": None,
                                    "completed_at": None,
                                    "duration_sec": 0,
                                }
                            )

        df = pd.DataFrame(matrix)
        causality_ablation_active = len(self.causality_ablations) > 1
        print(f"📊 Generated {len(df)} experimental runs")
        print(f"   - Claims: {len(self.claims)}")
        print(f"   - Models: {len(self.models)}")
        print(f"   - Planning ablations: {len(self.planning_ablations)}")
        print(
            f"   - Causality ablation: {'on/off for ' + str(self.causality_models) if causality_ablation_active else 'always on'}"
        )
        print(f"   - Replicates: {self.replicates}")

        return df

    def run_all_experiments(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        """Execute all runs with checkpoint/resume support."""
        results = []

        if self.use_docker:
            print("🐳 Starting backend in Docker...")
            self._start_docker_backend()
            time.sleep(5)  # Wait for backend to start

        total = len(matrix_df)
        for idx, row in matrix_df.iterrows():
            run_id = row["run_id"]
            print(
                f"\n[{idx+1}/{total}] Running: {row['claim_id']} | "
                f"Model: {row['model']} | "
                f"Planning: R={row['planning_reasoner']},C={row['planning_counter']} | "
                f"Causality: {row['causality_enabled']} | "
                f"Rep: {row['replicate']}"
            )

            try:
                started = datetime.now()
                response = self._execute_pipeline_run(row)

                completed = datetime.now()
                duration = (completed - started).total_seconds()

                # Extract metrics from response
                metrics = self._extract_metrics_from_response(response, row)
                metrics["run_id"] = run_id
                metrics["status"] = "completed"
                metrics["duration_sec"] = duration
                metrics["started_at"] = started.isoformat()
                metrics["completed_at"] = completed.isoformat()

                results.append(metrics)

                # Persist raw response
                self._persist_raw_response(run_id, response)

                print(f"   ✅ Completed in {duration:.1f}s")

            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}")
                results.append(
                    {
                        "run_id": run_id,
                        "claim_id": row["claim_id"],
                        "model": row["model"],
                        "planning_reasoner": row["planning_reasoner"],
                        "planning_counter": row["planning_counter"],
                        "causality_enabled": bool(row.get("causality_enabled", True)),
                        "replicate": row["replicate"],
                        "status": "failed",
                        "error": str(e)[:500],
                    }
                )

        return pd.DataFrame(results)

    def _execute_pipeline_run(self, row: dict) -> dict:
        """Execute a single pipeline run."""
        payload = {
            "claim": row["claim_text"],
            "include_precedents": True,
            "max_statutes": 100,
            "max_precedents": 5,
            "settings": {
                "reasoner_model": row["model"],
                "counter_model": row["model"],
                "reasoner_temperature": 0.0,
                "counter_temperature": 0.3,
                "llm_max_tokens": 7168,
                "enable_planning_reasoner": row["planning_reasoner"],
                "enable_planning_counter": row["planning_counter"],
                "reasoner_enable_causality": bool(row.get("causality_enabled", True)),
                "counter_enable_causality": bool(row.get("causality_enabled", True)),
            },
        }

        response = requests.post(
            f"{API_BASE}/pipeline",
            json=payload,
            timeout=1800,  # 30 min timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API returned {response.status_code}: {response.text[:200]}"
            )

        return response.json()

    def _extract_metrics_from_response(self, response: dict, row: dict) -> dict:
        """Extract relevant metrics from pipeline response."""
        metrics = {
            "claim_id": row["claim_id"],
            "domain": row["domain"],
            "model": row["model"],
            "planning_reasoner": row["planning_reasoner"],
            "planning_counter": row["planning_counter"],
            "causality_enabled": bool(row.get("causality_enabled", True)),
            "replicate": row["replicate"],
        }

        # AQA metrics (RQ1 — model efficacy)
        aqa_report = response.get("evaluation", {}).get("aqa_report", {})
        metrics["aqa_verdict"] = aqa_report.get("verdict", "unknown")
        net_plaus = aqa_report.get("net_plausibility", {})
        metrics["aqa_plausibility"] = float(net_plaus.get("final", 0.0))
        metrics["aqa_pro"] = float(net_plaus.get("pro", 0.0))
        metrics["aqa_contra"] = float(net_plaus.get("contra", 0.0))
        # AQA dimensional sub-scores
        scores = aqa_report.get("scores", {})
        metrics["aqa_cogency"] = float(scores.get("cogency", 0.0))
        metrics["aqa_norm_support"] = float(scores.get("norm_support", 0.0))
        metrics["aqa_semantics"] = float(scores.get("semantics", 0.0))
        metrics["aqa_attack_coverage"] = float(
            aqa_report.get("attack_coverage_score", 0.0)
        )

        # Citation faithfulness metrics (RQ2 — both Reasoner and Counter-Reasoner)
        consistency = response.get("evaluation", {}).get("consistency_report", {})
        # Counter-Reasoner citations
        ctr = consistency.get("counter_reasoner", {})
        metrics["citation_total"] = int(ctr.get("total_citations", 0))
        metrics["citation_valid"] = int(ctr.get("valid_citations", 0))
        metrics["citation_repaired"] = int(ctr.get("repaired_citations", 0))
        metrics["citation_dropped"] = int(ctr.get("dropped_citations", 0))
        metrics["citation_accuracy"] = metrics["citation_valid"] / max(
            1, metrics["citation_total"]
        )
        # Reasoner citations (faithfulness on the pro side)
        reas_c = consistency.get("reasoner", {})
        metrics["reasoner_citation_total"] = int(reas_c.get("total_citations", 0))
        metrics["reasoner_citation_valid"] = int(reas_c.get("valid_citations", 0))
        metrics["reasoner_citation_accuracy"] = metrics[
            "reasoner_citation_valid"
        ] / max(1, metrics["reasoner_citation_total"])

        # Planning / causality impact metrics (RQ3)
        token_stats = response.get("_token_stats", {})
        metrics["total_tokens"] = int(token_stats.get("total_completion_tokens", 0))
        metrics["reasoning_tokens"] = int(
            token_stats.get("reasoning_completion_tokens", 0)
        )
        metrics["counter_tokens"] = int(token_stats.get("counter_completion_tokens", 0))

        # Reasoning chain structure
        reasoner = response.get("reasoner", {})
        metrics["reasoning_steps"] = len(reasoner.get("reasoning_chain", []))
        metrics["reasoning_length"] = len(reasoner.get("raw_response", ""))

        counter = response.get("counter_reasoner", {})
        metrics["counter_steps"] = len(counter.get("reasoning_chain", []))
        metrics["counter_attacks_count"] = len(counter.get("selected_attack_ids", []))

        return metrics

    def _persist_raw_response(self, run_id: str, response: dict) -> None:
        """Save raw pipeline response for detailed analysis."""
        response_file = self.output_dir / "runs" / f"{run_id}.json"
        with open(response_file, "w") as f:
            json.dump(response, f, indent=2, default=str)

    def _start_docker_backend(self) -> None:
        """Start backend using Docker Compose (supports both modern and legacy CLI)."""
        # Prefer the modern `docker compose` plugin; fall back to `docker-compose`
        compose_cmds = [["docker", "compose"], ["docker-compose"]]
        last_error: str = ""

        for cmd_prefix in compose_cmds:
            try:
                subprocess.run(
                    [*cmd_prefix, "up", "-d"],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                print(f"✅ Docker backend started via {' '.join(cmd_prefix)}")
                break
            except FileNotFoundError:
                last_error = f"'{' '.join(cmd_prefix)}' not found"
                continue
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to start Docker: {e.stderr.decode()}")
        else:
            raise RuntimeError(f"Docker Compose not available: {last_error}")

        # Wait for health check
        for _ in range(60):
            try:
                response = requests.get(f"{API_BASE}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Backend health check passed")
                    return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError("Backend health check timeout after 120 seconds")

    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save results to CSV and JSON formats."""
        # CSV
        csv_path = self.output_dir / "metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n📊 Saved metrics to {csv_path}")

        # Parquet (if available)
        try:
            parquet_path = self.output_dir / "metrics.parquet"
            results_df.to_parquet(parquet_path, index=False)
            print(f"📊 Saved metrics to {parquet_path}")
        except ImportError:
            pass

        # Run matrix for reproducibility
        matrix_path = self.output_dir / "run_matrix.csv"
        results_df.to_csv(matrix_path, index=False)

    def run(self) -> None:
        """Execute complete DoE workflow."""
        print("=" * 70)
        print("🔬 MULTI-DoE FRAMEWORK - START")
        print("=" * 70)

        # Generate run matrix
        matrix_df = self.generate_run_matrix()
        matrix_df.to_csv(self.output_dir / "run_matrix.csv", index=False)

        # Execute runs
        results_df = self.run_all_experiments(matrix_df)

        # Save results
        self.save_results(results_df)

        print("\n" + "=" * 70)
        print("✅ MULTI-DoE FRAMEWORK - END")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Total runs: {len(results_df)}")
        print(f"Successful: {(results_df['status'] == 'completed').sum()}")
        print(f"Failed: {(results_df['status'] == 'failed').sum()}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-DoE framework for LexCausa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_multi_doe.py \\
    --claims-file claims.md \\
    --models gpt_oss_120b,groq_llama_scout_17b \\
    --domains CIVILE,PENALE \\
    --planning-ablations on,off \\
    --replicates 10 \\
    --out experiments/multi_doe/runs/$(date +%%Y%%m%%d_%%H%%M%%S)
        """,
    )

    parser.add_argument(
        "--claims-file",
        required=True,
        help="Path to claims.md file",
    )
    parser.add_argument(
        "--models",
        default="gpt_oss_120b,groq_llama_scout_17b",
        help="Comma-separated list of models to test",
    )
    parser.add_argument(
        "--domains",
        default="CIVILE,PENALE,AMMINISTRATIVO,MISTO",
        help="Comma-separated list of domains to include",
    )
    parser.add_argument(
        "--planning-ablations",
        default="on,off",
        choices=["on,off", "on", "off"],
        help="Planning ablation modes (on=both enabled, off=both disabled)",
    )
    parser.add_argument(
        "--causality-ablations",
        default="on",
        choices=["on,off", "on", "off"],
        help="Causality taxonomy ablation (on=always enabled; on,off=ablate for --causality-models)",
    )
    parser.add_argument(
        "--causality-models",
        default="gpt_oss_120b",
        help="Comma-separated models for which causality ablation is applied (default: gpt_oss_120b)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=10,
        help="Number of repetitions per condition",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Start backend using Docker Compose",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Parse arguments
    models = [m.strip() for m in args.models.split(",")]
    domains = [d.strip().upper() for d in args.domains.split(",")]
    planning_ablations = (
        [(True, True), (False, False)]
        if args.planning_ablations == "on,off"
        else [(True, True)] if args.planning_ablations == "on" else [(False, False)]
    )
    causality_ablations = (
        [True, False]
        if args.causality_ablations == "on,off"
        else [True] if args.causality_ablations == "on" else [False]
    )
    causality_models = [m.strip() for m in args.causality_models.split(",")]

    # Run DoE
    doe = MultiDoE(
        claims_file=args.claims_file,
        models=models,
        domains=domains,
        planning_ablations=planning_ablations,
        replicates=args.replicates,
        output_dir=args.out,
        use_docker=args.docker,
        seed=args.seed,
        causality_ablations=causality_ablations,
        causality_models=causality_models,
    )

    doe.run()


if __name__ == "__main__":
    main()
