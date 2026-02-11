"""
Scoring functions for AQA evaluation.

Provides argument quality, readability, coherence, norm support,
and semantics scoring for reasoning chain links.
"""

from __future__ import annotations

import sys
from pathlib import Path

import textstat

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore[assignment]


class ScoringMixin:
    """Mixin providing scoring methods to the evaluator."""

    def _get_arg_quality_model(self):
        if self._arg_quality is not None:
            return self._arg_quality
        if not self._aqa_arg_quality_use_model or not self._aqa_arg_quality_model:
            self._arg_quality = None
            return None
        if pipeline is None:
            self._arg_quality = None
            return None
        try:
            self._arg_quality = pipeline(
                "text-classification",
                model=self._aqa_arg_quality_model,
                top_k=None,
            )
        except Exception:
            self._arg_quality = None
        return self._arg_quality

    def _readability_score(self, text: str) -> float:
        text = self._normalize_text(text)
        if not text:
            return 0.0
        if textstat is not None:
            try:
                flesch = textstat.flesch_reading_ease(text)
                fog = textstat.gunning_fog(text)
                smog = textstat.smog_index(text)
                flesch_score = self._clamp01(flesch / 100.0)
                fog_score = 1.0 - self._clamp01(fog / 20.0)
                smog_score = 1.0 - self._clamp01(smog / 20.0)
                scores = [flesch_score, fog_score, smog_score]
                return sum(scores) / len(scores)
            except Exception:
                pass
        # Fallback: prefer moderate length
        word_count = len(text.split())
        if word_count <= 0:
            return 0.0
        if word_count <= 30:
            return 0.6
        if word_count <= 80:
            return 0.8
        if word_count <= 150:
            return 0.6
        return 0.4

    def _argument_quality_score(
        self, premise_text: str, rule_text: str, conclusion_text: str
    ) -> float:
        structure = 0
        structure += 1 if premise_text else 0
        structure += 1 if rule_text else 0
        structure += 1 if conclusion_text else 0
        structure_score = structure / 3.0
        quality_text = f"{premise_text}\n{rule_text}\n{conclusion_text}".strip()
        model_score = None
        model = self._get_arg_quality_model()
        if model is not None and quality_text:
            try:
                result = model(quality_text)
                scores = []
                for item in result[0] if isinstance(result, list) else []:
                    if isinstance(item, dict) and "score" in item:
                        scores.append(float(item["score"]))
                if scores:
                    model_score = max(scores)
            except Exception:
                model_score = None
        if model_score is not None:
            quality_score = self._clamp01(model_score)
        else:
            similarity_score = self._tfidf_similarity(
                premise_text + " " + rule_text, conclusion_text
            )
            if similarity_score == 0.0:
                similarity_score = self._similarity(
                    premise_text + " " + rule_text, conclusion_text
                )
            quality_score = self._clamp01(similarity_score)
        return self._clamp01(
            settings.readability_structure_weight * structure_score
            + settings.readability_quality_weight * quality_score
        )

    def _coherence_score(
        self, premise_text: str, conclusion_text: str, rule_text: str
    ) -> float:
        base = self._similarity(premise_text + " " + rule_text, conclusion_text)
        sentences = self._split_sentences(rule_text)
        if len(sentences) < 2:
            return base
        sims = []
        for idx in range(len(sentences) - 1):
            sims.append(self._similarity(sentences[idx], sentences[idx + 1]))
        if not sims:
            return base
        return self._clamp01(
            settings.coherence_base_weight * base
            + settings.coherence_chain_weight * (sum(sims) / len(sims))
        )

    def _build_chain_text(self, aspic_ir: dict) -> str:
        """
        Build the full text of a reasoning chain from ASPIC IR for norm support calculation.

        Extracts text from all arguments (premises, rules, conclusions) and reasoning chain steps.

        Args:
            aspic_ir: ASPIC IR structure

        Returns:
            Concatenated text of the entire chain.
        """
        if not aspic_ir:
            return ""

        parts = []

        # Extract from arguments
        for arg in aspic_ir.get("arguments", []):
            # Premises
            for premise in arg.get("premises", []):
                if isinstance(premise, dict):
                    parts.append(premise.get("text", ""))
                elif isinstance(premise, str):
                    parts.append(premise)
            # Rule/norm
            rule = arg.get("rule") or arg.get("norm") or {}
            if isinstance(rule, dict):
                parts.append(rule.get("text", ""))
            elif isinstance(rule, str):
                parts.append(rule)
            # Conclusion
            conclusion = arg.get("conclusion") or {}
            if isinstance(conclusion, dict):
                parts.append(conclusion.get("text", ""))
            elif isinstance(conclusion, str):
                parts.append(conclusion)

        # Extract from reasoning_chain steps
        for step in aspic_ir.get("reasoning_chain", []):
            if isinstance(step, dict):
                parts.append(step.get("text", ""))
            elif isinstance(step, str):
                parts.append(step)

        # Filter empty and join
        return " ".join(p for p in parts if p and isinstance(p, str))

    def _norm_support_score(
        self,
        text: str,
        retrieved_norms: list[dict] | None = None,
        max_citations: int | None = None,
    ) -> tuple[float, dict]:
        citations = self._extract_statute_citations(text)
        max_citations = max_citations or self._aqa_normsupport_max_citations
        citation_count = len(citations)
        citation_score = min(citation_count, max_citations) / max_citations
        retrieved_score = 0.0
        if retrieved_norms:
            scores = []
            for item in retrieved_norms:
                val = item.get("similarity") or item.get("score") or 0.0
                if isinstance(val, (int, float)):
                    scores.append(float(val))
            if scores:
                if self._aqa_normsupport_retrieved_agg == "max":
                    retrieved_score = max(scores)
                else:
                    retrieved_score = sum(scores) / len(scores)
        final_score = self._clamp01(
            self._aqa_normsupport_citation_weight * citation_score
            + self._aqa_normsupport_retrieved_weight * retrieved_score
        )
        details = {
            "citation_count": citation_count,
            "citation_score": citation_score,
            "retrieved_score": retrieved_score,
            "final": final_score,
        }
        return final_score, details

    def _semantics_score(
        self, premise_text: str, conclusion_text: str
    ) -> tuple[float, dict]:
        """Semantic coherence between premise and conclusion using cosine similarity."""
        score = self._similarity(premise_text, conclusion_text)
        return self._clamp01(score), {"method": "similarity", "score": score}
