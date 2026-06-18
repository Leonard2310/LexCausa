#!/usr/bin/env python
"""
Multi-DoE Analysis: statistical tests and reporting for the full-factorial
Reasoner x Counter-Reasoner DoE (Ibisco metrics format).

Implements the minimal test suite of the thesis (see Appendix "Statistical
Methods"):
- Factorial ANOVA: Reasoner, Counter, and planning main effects + the
  Reasoner x Counter interaction, with the eta-squared importance index. The
  Reasoner x Counter interaction is the thesis RQ2 (dialectical pairing).
- Model-class efficacy (RQ1): reasoning vs instruction-tuned, via a paired
  t-test by claim and Cohen's d.
- Citation faithfulness (RQ3): bootstrap confidence intervals.
- Planning ablation (RQ3): token / quality deltas; planning main effect in ANOVA.
- Auxiliary: token-cost efficiency.

Expects metrics.csv (Ibisco format) with columns including: reasoner_model,
counter_model, planning_reasoner, planning_counter, claim_id, aqa_plausibility,
citation_accuracy, citation_total, citation_repaired, total_tokens,
reasoning_tokens, counter_tokens, reasoning_steps, aqa_verdict, status, domain.

Usage:
    python scripts/analyze_multi_doe.py \\
        --run-dir experiments/multi_doe/runs/<ts> \\
        --output experiments/multi_doe/analysis/<ts>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Model aliases whose class is native reasoning (vs instruction-tuned).
REASONING_MODELS = {"deepseek_r1", "gpt_oss_120b"}


def _model_class(alias: str) -> str:
    return "reasoning" if alias in REASONING_MODELS else "instruction_tuned"


class AdvancedDoEAnalyzer:
    """Statistical analysis for the Reasoner x Counter Multi-DoE."""

    def __init__(self, run_dir: str, output_dir: str):
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = self.run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        self.results_df = pd.read_csv(metrics_path)

        df = self.results_df
        if "status" in df.columns:
            df = df[df["status"] == "completed"]
        df = df.copy()
        # Planning is toggled jointly on both agents; use it as a single factor.
        if "planning_reasoner" in df.columns:
            df["planning"] = df["planning_reasoner"].astype(bool)
        if "reasoner_model" in df.columns:
            df["reasoner_class"] = df["reasoner_model"].apply(_model_class)
        self.df = df

        print(f"✅ Loaded {len(self.results_df)} runs " f"({len(self.df)} completed)")

    def analyze(self) -> dict:
        """Run the full analysis."""
        return {
            "summary": self._summary_stats(),
            "anova": {
                "aqa_plausibility": self._factorial_anova("aqa_plausibility"),
                "citation_accuracy": self._factorial_anova("citation_accuracy"),
            },
            "model_class_efficacy": self._model_class_efficacy(),
            "citation_faithfulness": self._citation_faithfulness(),
            "planning_ablation": self._planning_ablation(),
            "token_cost_analysis": self._token_cost_analysis(),
        }

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def _summary_stats(self) -> dict:
        df = self.results_df
        completed = (
            int((df["status"] == "completed").sum())
            if "status" in df.columns
            else int(len(df))
        )
        failed = int((df["status"] == "failed").sum()) if "status" in df.columns else 0
        return {
            "total_runs": int(len(df)),
            "completed_runs": completed,
            "failed_runs": failed,
            "reasoner_models": sorted(
                self.df.get("reasoner_model", pd.Series()).unique().tolist()
            ),
            "counter_models": sorted(
                self.df.get("counter_model", pd.Series()).unique().tolist()
            ),
            "domains": sorted(self.df.get("domain", pd.Series()).unique().tolist()),
        }

    # ------------------------------------------------------------------ #
    # Factorial ANOVA (primary analysis)
    # ------------------------------------------------------------------ #
    def _factorial_anova(self, response: str) -> dict:
        """Factorial ANOVA on `response`.

        Main effects of Reasoner, Counter, and planning, plus the
        Reasoner x Counter interaction (RQ2). Manual sum-of-squares
        decomposition with p-values from the F-distribution; the eta-squared
        importance index (SS/SST) is the effect size. Variability from the
        remaining factors (claim, replicate, other interactions) is pooled into
        the error term. See Appendix "Statistical Methods".
        """
        df = self.df
        needed = {"reasoner_model", "counter_model", response}
        if not needed.issubset(df.columns):
            return {"error": f"Missing columns; need {sorted(needed)}"}

        cols = ["reasoner_model", "counter_model", response]
        has_planning = "planning" in df.columns
        if has_planning:
            cols.append("planning")
        d = df[cols].dropna()
        if d.empty:
            return {"error": "No data for ANOVA"}

        y = d[response].to_numpy(dtype=float)
        n_total = len(y)
        grand = float(y.mean())
        sst = float(((y - grand) ** 2).sum())
        if sst <= 0:
            return {"error": "Zero variance in response"}

        def _group_ss(by: Any) -> float:
            ss = 0.0
            for _, g in d.groupby(by):
                ss += len(g) * (float(g[response].mean()) - grand) ** 2
            return float(ss)

        a = int(d["reasoner_model"].nunique())
        b = int(d["counter_model"].nunique())
        ss_r = _group_ss("reasoner_model")
        ss_c = _group_ss("counter_model")
        ss_rc = _group_ss(["reasoner_model", "counter_model"]) - ss_r - ss_c

        effects: dict[str, tuple[float, int]] = {
            "reasoner": (ss_r, a - 1),
            "counter": (ss_c, b - 1),
            "reasoner_x_counter": (ss_rc, (a - 1) * (b - 1)),
        }
        model_df = (a - 1) + (b - 1) + (a - 1) * (b - 1)

        if has_planning:
            p = int(d["planning"].nunique())
            if p > 1:
                effects["planning"] = (_group_ss("planning"), p - 1)
                model_df += p - 1

        ss_e = sst - sum(ss for ss, _ in effects.values())
        df_e = n_total - 1 - model_df
        if df_e <= 0:
            return {"error": "Insufficient replication for ANOVA error df"}
        mse = ss_e / df_e

        def _effect(ss: float, dof: int) -> dict:
            ms = ss / dof if dof > 0 else 0.0
            f = ms / mse if mse > 0 else float("inf")
            return {
                "ss": ss,
                "df": dof,
                "ms": ms,
                "F": float(f),
                "p_value": float(stats.f.sf(f, dof, df_e)),
                "eta_squared": float(ss / sst),
            }

        return {
            "response": response,
            "grand_mean": grand,
            "sst": sst,
            "error": {"ss": ss_e, "df": df_e, "ms": mse},
            "effects": {k: _effect(ss, dof) for k, (ss, dof) in effects.items()},
            "assumptions": self._anova_assumptions(d, response),
        }

    @staticmethod
    def _anova_assumptions(d: pd.DataFrame, response: str) -> dict:
        """Check the parametric-ANOVA assumption of homoscedasticity (Levene's
        test across the Reasoner x Counter cells) and, when it fails, compute the
        non-parametric Kruskal-Wallis fallback on each main factor. Residual
        normality is assessed separately by the analyst (quantile-quantile plot).
        """
        cells = [
            g[response].to_numpy(dtype=float)
            for _, g in d.groupby(["reasoner_model", "counter_model"])
            if len(g) > 1
        ]
        if len(cells) < 2:
            return {}
        lev_stat, lev_p = stats.levene(*cells)
        homoscedastic = bool(lev_p > 0.05)
        out: dict[str, Any] = {
            "homoscedasticity_levene": {
                "statistic": float(lev_stat),
                "p_value": float(lev_p),
                "homoscedastic": homoscedastic,
            }
        }
        if not homoscedastic:

            def _kruskal(col: str) -> Any:
                groups = [
                    g[response].to_numpy(dtype=float)
                    for _, g in d.groupby(col)
                    if len(g) > 0
                ]
                if len(groups) < 2:
                    return None
                h_stat, h_p = stats.kruskal(*groups)
                return {"H": float(h_stat), "p_value": float(h_p)}

            fallback: dict[str, Any] = {
                "note": (
                    "Levene rejected equal variance; the parametric F-test is "
                    "replaced by Kruskal-Wallis on each multi-level factor and by "
                    "the Mann-Whitney U (Wilcoxon rank-sum) test on the two-level "
                    "planning factor."
                ),
                "reasoner": _kruskal("reasoner_model"),
                "counter": _kruskal("counter_model"),
            }
            if "planning" in d.columns and d["planning"].nunique() == 2:
                groups = [
                    g[response].to_numpy(dtype=float)
                    for _, g in d.groupby("planning")
                    if len(g) > 0
                ]
                if len(groups) == 2:
                    u_stat, u_p = stats.mannwhitneyu(
                        groups[0], groups[1], alternative="two-sided"
                    )
                    fallback["planning"] = {
                        "U": float(u_stat),
                        "p_value": float(u_p),
                    }
            out["non_parametric_fallback"] = fallback
        return out

    # ------------------------------------------------------------------ #
    # Model-class efficacy (RQ1): reasoning vs instruction-tuned
    # ------------------------------------------------------------------ #
    def _model_class_efficacy(self) -> dict:
        """Model-class efficacy (RQ1): native-reasoning vs instruction-tuned.

        Paired t-test by claim on AQA plausibility (each claim is evaluated under
        both classes) plus Cohen's d effect size.
        """
        df = self.df
        if df.empty or "reasoner_class" not in df.columns:
            return {"error": "No reasoner_class column / no data"}
        if df["reasoner_class"].nunique() < 2:
            return {"error": "Both model classes are required for the contrast"}

        out: dict[str, Any] = {}

        reasoning = df[df["reasoner_class"] == "reasoning"]
        instruct = df[df["reasoner_class"] == "instruction_tuned"]
        if "claim_id" in df.columns:
            r_by_claim = reasoning.groupby("claim_id")["aqa_plausibility"].mean()
            i_by_claim = instruct.groupby("claim_id")["aqa_plausibility"].mean()
            common = r_by_claim.index.intersection(i_by_claim.index)
            sr = r_by_claim.loc[common].to_numpy()
            si = i_by_claim.loc[common].to_numpy()
        else:
            sr = reasoning["aqa_plausibility"].to_numpy()
            si = instruct["aqa_plausibility"].to_numpy()

        if len(sr) > 1 and len(sr) == len(si):
            t_stat, p_value = stats.ttest_rel(sr, si)
            pooled_sd = np.sqrt(
                (
                    (len(sr) - 1) * sr.std(ddof=1) ** 2
                    + (len(si) - 1) * si.std(ddof=1) ** 2
                )
                / (len(sr) + len(si) - 2)
            )
            cohens_d = (sr.mean() - si.mean()) / pooled_sd if pooled_sd > 0 else 0.0
            out["aqa_plausibility_paired"] = {
                "reasoning_mean": float(sr.mean()),
                "instruction_tuned_mean": float(si.mean()),
                "n_claims": int(len(sr)),
                "test": "paired_t_by_claim",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "significant": bool(p_value < 0.05),
            }

        out["by_class"] = {
            cls: {
                "aqa_plausibility_mean": float(g["aqa_plausibility"].mean()),
                "reasoning_steps_mean": (
                    float(g["reasoning_steps"].mean())
                    if "reasoning_steps" in g.columns
                    else None
                ),
                "n": int(len(g)),
            }
            for cls, g in df.groupby("reasoner_class")
        }
        return out

    # ------------------------------------------------------------------ #
    # Citation faithfulness (RQ3): bootstrap CIs
    # ------------------------------------------------------------------ #
    def _citation_faithfulness(self) -> dict:
        df = self.df
        if df.empty or "citation_accuracy" not in df.columns:
            return {"error": "No citation_accuracy column / no data"}

        def _block(g: pd.DataFrame) -> dict:
            acc = g["citation_accuracy"].to_numpy(dtype=float)
            lo, hi = self._bootstrap_ci(acc)
            res = {
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                "accuracy_ci_95": [float(lo), float(hi)],
                "n": int(len(acc)),
            }
            if {"citation_repaired", "citation_total"}.issubset(g.columns):
                res["repair_rate"] = float(
                    g["citation_repaired"].sum() / max(1, g["citation_total"].sum())
                )
            return res

        out: dict[str, Any] = {"overall": _block(df)}
        if "reasoner_class" in df.columns:
            out["by_class"] = {
                cls: _block(g) for cls, g in df.groupby("reasoner_class")
            }
        return out

    # ------------------------------------------------------------------ #
    # Planning ablation (RQ3)
    # ------------------------------------------------------------------ #
    def _planning_ablation(self) -> dict:
        df = self.df
        if df.empty or "planning" not in df.columns:
            return {"error": "No planning column / no data"}

        on = df[df["planning"]]
        off = df[~df["planning"]]
        if on.empty or off.empty:
            return {"error": "Planning ablation conditions not found"}

        def _cell(g: pd.DataFrame) -> dict:
            return {
                "mean_tokens": float(g["total_tokens"].mean()),
                "median_tokens": float(g["total_tokens"].median()),
                "mean_reasoning_steps": float(g["reasoning_steps"].mean()),
                "mean_aqa_plausibility": float(g["aqa_plausibility"].mean()),
            }

        token_delta = float(on["total_tokens"].mean() - off["total_tokens"].mean())
        off_tokens = float(off["total_tokens"].mean())
        return {
            "planning_on": _cell(on),
            "planning_off": _cell(off),
            "token_delta": {
                "absolute": token_delta,
                "percentage": (
                    float(token_delta / off_tokens * 100) if off_tokens else 0.0
                ),
                "planning_on_cheaper": token_delta < 0,
            },
            "reasoning_steps_delta": float(
                on["reasoning_steps"].mean() - off["reasoning_steps"].mean()
            ),
            "quality_delta": float(
                on["aqa_plausibility"].mean() - off["aqa_plausibility"].mean()
            ),
        }

    # ------------------------------------------------------------------ #
    # Token cost
    # ------------------------------------------------------------------ #
    def _token_cost_analysis(self) -> dict:
        df = self.df
        if df.empty or "total_tokens" not in df.columns:
            return {"error": "No token data"}

        group_col = (
            "reasoner_class" if "reasoner_class" in df.columns else "reasoner_model"
        )
        out: dict[str, Any] = {}
        for key, g in df.groupby(group_col):
            mean_tokens = float(g["total_tokens"].mean())
            block = {
                "tokens_per_run_mean": mean_tokens,
                "tokens_per_run_median": float(g["total_tokens"].median()),
                "tokens_per_quality_point": float(
                    mean_tokens / max(0.01, g["aqa_plausibility"].mean())
                ),
            }
            if "reasoning_tokens" in g.columns:
                block["reasoning_tokens_ratio"] = float(
                    g["reasoning_tokens"].mean() / max(1.0, mean_tokens)
                )
            out[str(key)] = block
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bootstrap_ci(
        values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
    ) -> tuple:
        """Percentile bootstrap confidence interval of the mean."""
        if len(values) == 0:
            return (float("nan"), float("nan"))
        resamples = np.random.choice(
            values, size=(n_bootstrap, len(values)), replace=True
        )
        means = np.mean(resamples, axis=1)
        lower = float(np.percentile(means, (1 - ci) / 2 * 100))
        upper = float(np.percentile(means, (1 + ci) / 2 * 100))
        return lower, upper

    def save_report(self, analysis: dict) -> None:
        """Save the analysis as JSON and print a short summary."""
        report_path = self.output_dir / "doe_analysis.json"
        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n📊 Analysis report saved to {report_path}")

        print("\n" + "=" * 70)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 70)
        print(json.dumps(analysis.get("summary", {}), indent=2))

        anova = analysis.get("anova", {}).get("aqa_plausibility", {})
        if "effects" in anova:
            print("\n🎯 Factorial ANOVA (AQA plausibility):")
            for name, e in anova["effects"].items():
                print(
                    f"   {name:20s} F={e['F']:.2f} p={e['p_value']:.2e} "
                    f"eta^2={e['eta_squared']:.1%}"
                )

        rq1 = analysis.get("model_class_efficacy", {}).get("aqa_plausibility_paired")
        if rq1:
            print("\n🎯 RQ1 (reasoning vs instruction-tuned, paired by claim):")
            print(f"   reasoning mean:        {rq1['reasoning_mean']:.3f}")
            print(f"   instruction-tuned mean: {rq1['instruction_tuned_mean']:.3f}")
            print(f"   p={rq1['p_value']:.2e}  d={rq1['cohens_d']:.2f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze Multi-DoE results")
    parser.add_argument("--run-dir", required=True, help="DoE run directory")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    args = parser.parse_args()

    analyzer = AdvancedDoEAnalyzer(run_dir=args.run_dir, output_dir=args.output)
    analysis = analyzer.analyze()
    analyzer.save_report(analysis)
    print("✅ Analysis complete")


if __name__ == "__main__":
    main()
