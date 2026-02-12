"""
LexCausa Polisher-Evaluator Agent.

The Polisher-Evaluator is responsible for:
1. Receiving arguments from Reasoner and Counter-Reasoner
2. Evaluating the dialectical exchange
3. Checking consistency of reasoning chains against the knowledge base
4. Determining which arguments prevail via AQA scoring
5. Repairing and polishing the final output for presentation

This agent acts as a judge/evaluator of the argumentation.

The implementation is split across several mixin modules under
``agents.evaluation``:

- :class:`~agents.evaluation.nlp_utils.NLPUtils` -- text helpers
- :class:`~agents.evaluation.scoring.ScoringMixin` -- quality scoring
- :class:`~agents.evaluation.consistency_checker.ConsistencyMixin` -- citation
  verification & chain repair
- :class:`~agents.evaluation.aqa_engine.AQAEngineMixin` -- ASPIC+ AQA pipeline
"""

import re
from typing import Any, Optional

from config import settings

from .base import AgentConfig, BaseAgent
from .evaluation import AQAEngineMixin, ConsistencyMixin, NLPUtils, ScoringMixin
from .evaluation.models import (  # noqa: F401 -- re-exported
    CitationCheck,
    ConsistencyReport,
    EvaluationResult,
    MismatchAction,
)

# ---------------------------------------------------------------------------
# Main class -- composes all mixins
# ---------------------------------------------------------------------------


class PolisherEvaluator(
    NLPUtils,
    ScoringMixin,
    ConsistencyMixin,
    AQAEngineMixin,
    BaseAgent,
):
    """
    Legal Polisher-Evaluator Agent.

    Evaluates the dialectical exchange between Reasoner and Counter-Reasoner,
    determines which arguments prevail, and produces a polished final output.

    Implementation is split across mixin classes for maintainability:
    - **NLPUtils**: text normalization, embedding, similarity
    - **ScoringMixin**: argument quality, coherence, readability
    - **ConsistencyMixin**: citation verification, mismatch repair
    - **AQAEngineMixin**: ASPIC+ link building, cross-attack, verdict
    """

    STATUTE_PATTERN = re.compile(
        r"(?:art(?:icolo)?\.?\s*)(\d{1,4})\s*"
        r"(c\.?[cp]\.?|cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?))?",
        re.IGNORECASE,
    )
    PRECEDENT_PATTERN = re.compile(
        r"(?:Cass(?:azione)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?)?\.?\s*"
        r"(?:n\.?\s*)?(\d+)(?:/(\d{4}))?)|"
        r"(?:sentenza\s+n\.?\s*(\d+)(?:/(\d{4}))?)",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Polisher-Evaluator agent."""
        super().__init__(config)
        self._log("Polisher-Evaluator initialized")
        self._embedder = None
        self._arg_quality = None
        self._tfidf_vectorizer = None
        self._tfidf_cache: dict[str, Any] = {}
        self._embed_cache: dict[str, Any] = {}
        self._statute_meta_cache: dict[tuple[str, str], dict] = {}
        self._aqa_enabled = settings.aqa_enabled
        self._aqa_alpha = settings.aqa_alpha
        self._aqa_beta = settings.aqa_beta
        self._aqa_gamma = settings.aqa_gamma
        self._aqa_attack_top_k = settings.aqa_attack_top_k
        self._aqa_min_semantic_overlap = settings.aqa_min_semantic_overlap
        self._aqa_min_strength_ratio = settings.aqa_min_strength_ratio
        self._aqa_damage_factor = settings.aqa_damage_factor
        self._aqa_allow_factual_attacks = settings.aqa_allow_factual_attacks
        self._aqa_allow_cross_codice = settings.aqa_allow_cross_codice

        self._aqa_attack_type_multipliers = settings.aqa_attack_type_multipliers
        self._aqa_strength_ratio_by_type = settings.aqa_strength_ratio_by_type
        self._aqa_severity_book_map = settings.aqa_severity_book_map
        self._aqa_verdict_pos = settings.aqa_verdict_pos_threshold
        self._aqa_verdict_neg = settings.aqa_verdict_neg_threshold
        self._aqa_embedding_model = settings.aqa_embedding_model
        self._aqa_arg_quality_model = settings.aqa_argument_quality_model
        self._aqa_arg_quality_use_model = settings.aqa_argument_quality_use_model
        self._aqa_tfidf_max_features = settings.aqa_tfidf_max_features
        self._aqa_normsupport_max_citations = settings.aqa_normsupport_max_citations
        self._aqa_normsupport_citation_weight = settings.aqa_normsupport_citation_weight
        self._aqa_normsupport_retrieved_weight = (
            settings.aqa_normsupport_retrieved_weight
        )
        self._aqa_normsupport_retrieved_agg = settings.aqa_normsupport_retrieved_agg
        self._aqa_severity_map_penale = settings.aqa_severity_map_penale
        self._aqa_severity_map_civile = settings.aqa_severity_map_civile

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        claim: str,
        domain: str = "CIVILE",
        reasoner_output: dict | None = None,
        counter_reasoner_output: dict | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the dialectical exchange and produce final output."""
        self._log("Starting consistency evaluation...")
        self._log(f"Domain: {domain}")

        # Refresh AQA parameters (supports frontend overrides)
        self._aqa_alpha = settings.aqa_alpha
        self._aqa_beta = settings.aqa_beta
        self._aqa_gamma = settings.aqa_gamma
        self._aqa_min_semantic_overlap = settings.aqa_min_semantic_overlap
        self._aqa_min_strength_ratio = settings.aqa_min_strength_ratio
        self._aqa_damage_factor = settings.aqa_damage_factor
        self._aqa_allow_factual_attacks = settings.aqa_allow_factual_attacks
        self._aqa_allow_cross_codice = settings.aqa_allow_cross_codice
        self._aqa_strength_ratio_by_type = settings.aqa_strength_ratio_by_type

        reasoner_output = reasoner_output or {}
        counter_reasoner_output = counter_reasoner_output or {}

        # ----- Consistency checking -----
        reasoner_chain = reasoner_output.get("reasoning_chain", [])
        reasoner_raw = reasoner_output.get("raw_response", "")
        reasoner_aspic_ir = reasoner_output.get("aspic_ir", {})
        reasoner_report = self._check_consistency(
            agent="reasoner",
            reasoning_chain=reasoner_chain,
            raw_response=reasoner_raw,
            domain=domain,
            aspic_ir=reasoner_aspic_ir,
        )

        counter_chain = counter_reasoner_output.get("reasoning_chain", [])
        counter_raw = counter_reasoner_output.get("raw_response", "")
        counter_aspic_ir = counter_reasoner_output.get("aspic_ir", {})
        counter_report = self._check_consistency(
            agent="counter_reasoner",
            reasoning_chain=counter_chain,
            raw_response=counter_raw,
            domain=domain,
            aspic_ir=counter_aspic_ir,
        )

        self._log(
            f"Reasoner consistency: {reasoner_report.consistency_score:.2f} "
            f"({reasoner_report.valid_citations}/"
            f"{reasoner_report.total_citations} valid)"
        )
        self._log(
            f"Counter consistency: {counter_report.consistency_score:.2f} "
            f"({counter_report.valid_citations}/"
            f"{counter_report.total_citations} valid)"
        )

        # ----- Repair chains if needed -----
        self._log("Checking if reasoning chains need repair...")

        if (
            reasoner_report.repaired_citations > 0
            or reasoner_report.dropped_citations > 0
        ):
            repaired_reasoner_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=reasoner_raw,
                citation_checks=reasoner_report.citation_checks,
                agent_name="reasoner",
            )
        else:
            repaired_reasoner_chain = reasoner_raw

        if (
            counter_report.repaired_citations > 0
            or counter_report.dropped_citations > 0
        ):
            repaired_counter_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=counter_raw,
                citation_checks=counter_report.citation_checks,
                agent_name="counter_reasoner",
            )
        else:
            repaired_counter_chain = counter_raw

        # Repair ASPIC IR structures
        repaired_reasoner_aspic = self._repair_aspic_ir(
            aspic_ir=reasoner_aspic_ir,
            citation_checks=reasoner_report.citation_checks,
            repaired_chain_text=repaired_reasoner_chain,
            claim=claim,
            role="support",
            statutes=reasoner_output.get("relevant_statutes", []),
            precedents=reasoner_output.get("relevant_precedents", []),
            metadata=reasoner_aspic_ir.get("metadata", {}),
        )
        repaired_counter_aspic = self._repair_aspic_ir(
            aspic_ir=counter_aspic_ir,
            citation_checks=counter_report.citation_checks,
            repaired_chain_text=repaired_counter_chain,
            claim=claim,
            role="counter",
            statutes=counter_reasoner_output.get("relevant_statutes", []),
            precedents=counter_reasoner_output.get("relevant_precedents", []),
            metadata=counter_aspic_ir.get("metadata", {}),
        )

        # Dialectical tree bundle
        reasoner_ir = reasoner_output.get("aspic_ir")
        counter_ir = counter_reasoner_output.get("aspic_ir")
        dialectical_tree = {}
        if reasoner_ir or counter_ir:
            dialectical_tree = {
                "schema": "aspic_ir_bundle_v1",
                "reasoner": reasoner_ir,
                "counter": counter_ir,
                "repaired_reasoner": repaired_reasoner_aspic,
                "repaired_counter": repaired_counter_aspic,
            }

        summary = self._generate_consistency_summary(reasoner_report, counter_report)

        # ----- AQA Phase -----
        aqa_reasoner_ir = (
            repaired_reasoner_aspic
            if repaired_reasoner_aspic
            else (reasoner_aspic_ir or {})
        )
        aqa_counter_ir = (
            repaired_counter_aspic
            if repaired_counter_aspic
            else (counter_aspic_ir or {})
        )

        if repaired_reasoner_aspic:
            r_meta = repaired_reasoner_aspic.get("_repair_metadata", {})
            self._log(
                f"AQA using REPAIRED Reasoner IR "
                f"(repaired={r_meta.get('total_repaired', 0)}, "
                f"dropped={r_meta.get('total_dropped', 0)})"
            )
        else:
            self._log("AQA using ORIGINAL Reasoner IR (no repairs needed)")

        if repaired_counter_aspic:
            c_meta = repaired_counter_aspic.get("_repair_metadata", {})
            self._log(
                f"AQA using REPAIRED Counter IR "
                f"(repaired={c_meta.get('total_repaired', 0)}, "
                f"dropped={c_meta.get('total_dropped', 0)})"
            )
        else:
            self._log("AQA using ORIGINAL Counter IR (no repairs needed)")

        aqa_report = self._run_aqa_phase(
            reasoner_ir=aqa_reasoner_ir,
            counter_ir=aqa_counter_ir,
            domain=domain,
        )

        self._log("Evaluation complete")

        return EvaluationResult(
            claim=claim,
            winning_side="",
            confidence=0.0,
            consistency_report={
                "reasoner": reasoner_report.to_dict(),
                "counter_reasoner": counter_report.to_dict(),
            },
            aqa_report=aqa_report,
            summary=summary,
            polished_response="",
            dialectical_tree=dialectical_tree,
            repaired_reasoner_chain=repaired_reasoner_chain,
            repaired_counter_chain=repaired_counter_chain,
            repaired_reasoner_aspic_ir=repaired_reasoner_aspic,
            repaired_counter_aspic_ir=repaired_counter_aspic,
        )
