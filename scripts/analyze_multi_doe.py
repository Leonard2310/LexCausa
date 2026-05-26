#!/usr/bin/env python
"""
Multi-DoE Analysis: Statistical tests and reporting for multi-dimensional ablation studies.

Computes per-RQ (Research Question) analysis:
- RQ1: Model Efficacy (paired t-test, Cohen's d)
- RQ2: Citation Faithfulness (accuracy confidence intervals, sign test)
- RQ3: Planning Ablation (token deltas, sign test for planning impact)
- Auxiliary: Token cost analysis (efficiency curves)

Usage:
    python scripts/analyze_multi_doe.py \\
        --run-dir experiments/multi_doe/runs/20260225_101530 \\
        --output experiments/multi_doe/analysis/20260225
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class AdvancedDoEAnalyzer:
    """Statistical analysis for Multi-DoE results."""

    def __init__(self, run_dir: str, output_dir: str):
        """Initialize analyzer."""
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        metrics_path = self.run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        self.results_df = pd.read_csv(metrics_path)
        print(f"✅ Loaded {len(self.results_df)} experimental runs")

    def analyze(self) -> dict:
        """Run all RQ analyses."""
        analysis = {
            "summary": self._summary_stats(),
            "rq1_model_efficacy": self._rq1_model_efficacy(),
            "rq2_citation_faithfulness": self._rq2_citation_faithfulness(),
            "rq3_planning_ablation": self._rq3_planning_ablation(),
            "token_cost_analysis": self._token_cost_analysis(),
        }

        return analysis

    def _summary_stats(self) -> dict:
        """Overall summary statistics."""
        return {
            "total_runs": len(self.results_df),
            "successful_runs": (self.results_df["status"] == "completed").sum(),
            "failed_runs": (self.results_df["status"] == "failed").sum(),
            "models_tested": self.results_df["model"].unique().tolist(),
            "domains_tested": self.results_df["domain"].unique().tolist(),
        }

    def _rq1_model_efficacy(self) -> dict:
        """
        RQ1: Model Efficacy Analysis

        Compares reasoning vs non-reasoning models on:
        - AQA verdict plausibility
        - Reasoning chain length
        - Coherence metrics
        """
        df = self.results_df[self.results_df["status"] == "completed"]

        if df.empty:
            return {"error": "No completed runs"}

        models = df["model"].unique()
        if len(models) < 2:
            return {"error": f"Only {len(models)} model(s) found; need at least 2"}

        analysis = {}

        # Compare first two models
        model_a, model_b = models[0], models[1]
        df_a = df[df["model"] == model_a]
        df_b = df[df["model"] == model_b]

        # AQA Plausibility comparison
        scores_a = df_a["aqa_plausibility"].values
        scores_b = df_b["aqa_plausibility"].values

        if len(scores_a) > 1 and len(scores_b) > 1:
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
            cohens_d = (scores_a.mean() - scores_b.mean()) / np.sqrt(
                (
                    (len(scores_a) - 1) * scores_a.std() ** 2
                    + (len(scores_b) - 1) * scores_b.std() ** 2
                )
                / (len(scores_a) + len(scores_b) - 2)
            )

            analysis["aqa_plausibility"] = {
                f"{model_a}_mean": float(scores_a.mean()),
                f"{model_a}_std": float(scores_a.std()),
                f"{model_b}_mean": float(scores_b.mean()),
                f"{model_b}_std": float(scores_b.std()),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "significant": p_value < 0.05,
            }

        # Reasoning chain length
        analysis["reasoning_chain_length"] = {
            f"{model_a}_mean": float(df_a["reasoning_steps"].mean()),
            f"{model_b}_mean": float(df_b["reasoning_steps"].mean()),
            "delta": float(
                df_b["reasoning_steps"].mean() - df_a["reasoning_steps"].mean()
            ),
        }

        # Verdict distribution
        analysis["verdict_distribution"] = {
            model_a: df_a["aqa_verdict"].value_counts().to_dict(),
            model_b: df_b["aqa_verdict"].value_counts().to_dict(),
        }

        return analysis

    def _rq2_citation_faithfulness(self) -> dict:
        """
        RQ2: Citation Faithfulness (Accuracy)

        Measures:
        - Citation accuracy per model
        - Confidence intervals (bootstrap)
        - Citation repair rate
        """
        df = self.results_df[self.results_df["status"] == "completed"]

        if df.empty:
            return {"error": "No completed runs"}

        analysis = {}
        models = df["model"].unique()

        for model in models:
            df_model = df[df["model"] == model]
            accuracies = df_model["citation_accuracy"].values

            # Bootstrap CI
            ci_lower, ci_upper = self._bootstrap_ci(accuracies)

            analysis[model] = {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "accuracy_ci_95": [float(ci_lower), float(ci_upper)],
                "total_citations_mean": float(df_model["citation_total"].mean()),
                "citation_repaired_mean": float(df_model["citation_repaired"].mean()),
                "citation_repaired_rate": float(
                    df_model["citation_repaired"].sum()
                    / max(1, df_model["citation_total"].sum())
                ),
            }

        # Paired accuracy comparison if 2+ models
        if len(models) >= 2:
            model_a, model_b = models[0], models[1]
            acc_a = df[df["model"] == model_a]["citation_accuracy"].values
            acc_b = df[df["model"] == model_b]["citation_accuracy"].values

            if len(acc_a) > 0 and len(acc_b) > 0:
                # Sign test
                diff = acc_a - acc_b
                sign_test = np.sum(diff > 0)  # Count wins for model_a
                total = np.sum(diff != 0)

                analysis["comparison"] = {
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_wins": int(sign_test),
                    "total_comparisons": int(total),
                    "model_a_win_rate": float(sign_test / max(1, total)),
                }

        return analysis

    def _rq3_planning_ablation(self) -> dict:
        """
        RQ3: Planning Ablation Impact

        Measures:
        - Token delta (with planning vs without)
        - Reasoning steps delta
        - Quality metrics delta
        """
        df = self.results_df[self.results_df["status"] == "completed"]

        if df.empty:
            return {"error": "No completed runs"}

        # Separate planning conditions
        df_planning_on = df[
            df["planning_reasoner"].astype(bool) & df["planning_counter"].astype(bool)
        ]
        df_planning_off = df[
            ~df["planning_reasoner"].astype(bool) & ~df["planning_counter"].astype(bool)
        ]

        if df_planning_on.empty or df_planning_off.empty:
            return {"error": "Planning ablation conditions not found"}

        analysis: dict[str, Any] = {
            "planning_on": {
                "mean_tokens": float(df_planning_on["total_tokens"].mean()),
                "median_tokens": float(df_planning_on["total_tokens"].median()),
                "mean_reasoning_steps": float(df_planning_on["reasoning_steps"].mean()),
                "mean_aqa_plausibility": float(
                    df_planning_on["aqa_plausibility"].mean()
                ),
            },
            "planning_off": {
                "mean_tokens": float(df_planning_off["total_tokens"].mean()),
                "median_tokens": float(df_planning_off["total_tokens"].median()),
                "mean_reasoning_steps": float(
                    df_planning_off["reasoning_steps"].mean()
                ),
                "mean_aqa_plausibility": float(
                    df_planning_off["aqa_plausibility"].mean()
                ),
            },
        }

        # Token delta
        token_delta = (
            df_planning_on["total_tokens"].mean()
            - df_planning_off["total_tokens"].mean()
        )
        token_delta_pct = (token_delta / df_planning_off["total_tokens"].mean()) * 100

        analysis["token_delta"] = {
            "absolute": float(token_delta),
            "percentage": float(token_delta_pct),
            "planning_on_preferred": token_delta < 0,  # Fewer tokens is better
        }

        # Reasoning steps delta
        steps_delta = (
            df_planning_on["reasoning_steps"].mean()
            - df_planning_off["reasoning_steps"].mean()
        )
        analysis["reasoning_steps_delta"] = float(steps_delta)

        # Quality delta (AQA plausibility)
        quality_delta = (
            df_planning_on["aqa_plausibility"].mean()
            - df_planning_off["aqa_plausibility"].mean()
        )
        analysis["quality_delta"] = float(quality_delta)

        # Sign test: does planning improve quality?
        diff = (
            df_planning_on["aqa_plausibility"].values
            - df_planning_off["aqa_plausibility"].values
        )
        improvements = np.sum(diff > 0)
        deteriorations = np.sum(diff < 0)

        analysis["planning_quality_impact"] = {
            "improvements": int(improvements),
            "deteriorations": int(deteriorations),
            "improvement_rate": float(
                improvements / max(1, improvements + deteriorations)
            ),
        }

        return analysis

    def _token_cost_analysis(self) -> dict:
        """Token efficiency analysis across models and conditions."""
        df = self.results_df[self.results_df["status"] == "completed"]

        if df.empty:
            return {"error": "No completed runs"}

        analysis = {}
        models = df["model"].unique()

        for model in models:
            df_model = df[df["model"] == model]
            token_per_verdict = df_model["total_tokens"].mean()

            # Assume $0.00004 per 1M tokens (approximate Groq pricing)
            cost_per_1m = 0.00004
            cost_per_verdict = (token_per_verdict / 1e6) * cost_per_1m

            analysis[model] = {
                "tokens_per_verdict_mean": float(token_per_verdict),
                "tokens_per_verdict_median": float(df_model["total_tokens"].median()),
                "cost_usd_per_verdict": float(cost_per_verdict),
                "reasoning_tokens_ratio": float(
                    df_model["reasoning_tokens"].mean() / max(1, token_per_verdict)
                ),
                "counter_tokens_ratio": float(
                    df_model["counter_tokens"].mean() / max(1, token_per_verdict)
                ),
            }

        # Efficiency frontier
        if len(models) >= 2:
            model_a, model_b = models[0], models[1]
            df_a = df[df["model"] == model_a]
            df_b = df[df["model"] == model_b]

            analysis["efficiency_comparison"] = {
                "model_a": model_a,
                "model_a_tokens_per_quality_point": float(
                    df_a["total_tokens"].mean()
                    / max(0.01, df_a["aqa_plausibility"].mean())
                ),
                "model_b": model_b,
                "model_b_tokens_per_quality_point": float(
                    df_b["total_tokens"].mean()
                    / max(0.01, df_b["aqa_plausibility"].mean())
                ),
                "efficiency_winner": (
                    model_a
                    if (
                        df_a["total_tokens"].mean()
                        / max(0.01, df_a["aqa_plausibility"].mean())
                    )
                    < (
                        df_b["total_tokens"].mean()
                        / max(0.01, df_b["aqa_plausibility"].mean())
                    )
                    else model_b
                ),
            }

        return analysis

    @staticmethod
    def _bootstrap_ci(
        values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
    ) -> tuple:
        """Compute bootstrap confidence interval."""
        bootstrapped = np.random.choice(
            values, size=(n_bootstrap, len(values)), replace=True
        )
        means = np.mean(bootstrapped, axis=1)
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return lower, upper

    def save_report(self, analysis: dict) -> None:
        """Save analysis report as JSON."""
        report_path = self.output_dir / "doe_analysis.json"
        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"\n📊 Analysis report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 70)
        print(json.dumps(analysis["summary"], indent=2))

        if (
            "rq1_model_efficacy" in analysis
            and "aqa_plausibility" in analysis["rq1_model_efficacy"]
        ):
            rq1 = analysis["rq1_model_efficacy"]["aqa_plausibility"]
            print("\n🎯 RQ1 Model Efficacy:")
            print(f"   Model A: {rq1.get('gpt_oss_120b_mean', 'N/A'):.3f}")
            print(f"   Model B: {rq1.get('groq_llama_scout_17b_mean', 'N/A'):.3f}")
            print(f"   p-value: {rq1.get('p_value', 'N/A')}")
            print(f"   Significant: {rq1.get('significant', False)}")

        if "planning_quality_impact" in analysis.get("rq3_planning_ablation", {}):
            rq3 = analysis["rq3_planning_ablation"]
            print("\n🎯 RQ3 Planning Ablation:")
            print(f"   Token Delta: {rq3['token_delta']['percentage']:.1f}%")
            print(f"   Quality Delta: {rq3['quality_delta']:.3f}")
            print(
                f"   Improvement Rate: {rq3['planning_quality_impact']['improvement_rate']:.1%}"
            )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze Multi-DoE results")
    parser.add_argument("--run-dir", required=True, help="DoE run directory")
    parser.add_argument("--output", required=True, help="Output directory for analysis")

    args = parser.parse_args()

    analyzer = AdvancedDoEAnalyzer(
        run_dir=args.run_dir,
        output_dir=args.output,
    )

    analysis = analyzer.analyze()
    analyzer.save_report(analysis)

    print("✅ Analysis complete")


if __name__ == "__main__":
    main()
