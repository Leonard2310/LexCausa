"""
LexCausa Reasoner Agent.

The Reasoner is responsible for:
1. Receiving a legal claim with pre-retrieved statutes and precedents
2. Generating a primary legal reasoning based on the provided knowledge base
3. Building a reasoning chain that connects the claim to legal norms

Uses LangGraph with Groq Cloud for LLM-powered reasoning.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .aspic_formatter import AspicFormatter
from .base import AgentConfig, BaseAgent
from .citation_utils import (
    extract_article_mentions,
    format_article_citation,
    normalize_article_id,
)
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_statute_by_article_tool
from .tools.prompt_registry import get_prompt, render_prompt
from .tools.taxonomy_tools import get_causality_theory_tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_react_invoke  # noqa: E402
from services.pipeline_control import PipelineCancelled  # noqa: E402

# System prompt for the Reasoner (with pre-retrieved context)
REASONER_SYSTEM_PROMPT = get_prompt("reasoner.system")


@dataclass
class ReasonerOutput:
    """Structured output from the Reasoner."""

    claim: str
    causality_classification: dict
    causal_type_id: str = ""
    theory_id: str = ""
    causal_type_ids_for_counter: list[str] = field(default_factory=list)
    mismatch_status: str = ""
    anchor_norms: dict = field(default_factory=dict)
    principle_tests: list[dict] = field(default_factory=list)
    relevant_statutes: list[dict] = field(default_factory=list)
    relevant_precedents: list[dict] = field(default_factory=list)
    arguments: list[dict] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)
    raw_response: str = ""
    conclusion: str = ""
    aspic_ir: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causality": self.causality_classification,
            "causal_type_id": self.causal_type_id,
            "theory_id": self.theory_id,
            "causal_type_ids_for_counter": self.causal_type_ids_for_counter,
            "mismatch_status": self.mismatch_status,
            "anchor_norms": self.anchor_norms,
            "principle_tests": self.principle_tests,
            "statutes": self.relevant_statutes,
            "precedents": self.relevant_precedents,
            "arguments": self.arguments,
            "reasoning_chain": self.reasoning_chain,
            "raw_response": self.raw_response,
            "conclusion": self.conclusion,
            "aspic_ir": self.aspic_ir,
        }


class Reasoner(BaseAgent):
    """
    Legal Reasoner Agent.

    Analyzes legal claims using pre-retrieved knowledge (statutes/precedents),
    classifies causality type, and generates the primary reasoning chain.

    Flow:
    1. api_server pre-retrieves statutes and precedents
    2. api_server filters statutes using filter_irrelevant_statutes()
    3. Reasoner.run() receives the filtered knowledge base
    4. ReAct agent uses tools (classify_causality, get_causality_theory)
       to build arguments based on the provided knowledge
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Reasoner agent."""
        super().__init__(config)
        self._react_agent = None
        self._max_plan_retries = 3
        self._max_step_rewrites = 2

    def _resilient_model_order(self) -> list[str] | None:
        """Reasoner fallback chain from settings (selected model first)."""
        preferred_chain = settings.reasoner_model_fallback_order
        selected = settings.resolve_model_name(self.config.model_name)
        order = [selected] + [m for m in preferred_chain if m != selected]
        return order

    @property
    def tools(self) -> list:
        """
        Get the tools available to this agent.

        NOTE: No search tools - the agent works with pre-retrieved context.
        Tools are limited to statute lookup to keep the chain deterministic.
        """
        return [
            get_statute_by_article_tool,  # For looking up specific articles by number
        ]

    @property
    def react_agent(self):
        """Lazy initialization of the ReAct agent using LangGraph."""
        if self._react_agent is None:
            self._react_agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=REASONER_SYSTEM_PROMPT,
            )
        return self._react_agent

    def _build_react_agent(self, api_key: str, model: str):
        """Build a fresh ReAct agent with specified key and model (for resilient invocation)."""
        llm = get_chat_groq(
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=api_key,
        )
        return create_react_agent(
            llm,
            self.tools,
            prompt=REASONER_SYSTEM_PROMPT,
        )

    def run(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        pre_retrieved_statutes: list[dict],
        pre_retrieved_precedents: list[dict],
        enable_causality: bool = True,
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> ReasonerOutput:
        """
        Two-phase reasoning:
        1) Generate initial reasoning from claim + retrieved sources.
        2) Classify causality on that reasoning (not on the claim), validate vs router claim-class.
           If validated, inject anchor norms/principle tests and refine reasoning (with cross-ref expansion).
        """
        self._log(f"Analyzing claim: {claim[:100]}...")
        self._log(
            f"📚 Knowledge base: {len(pre_retrieved_statutes)} statutes, {len(pre_retrieved_precedents)} precedents"
        )

        if not routing_decision or not routing_decision.domain:
            raise ValueError("routing_decision with a valid domain is required")

        if not pre_retrieved_statutes and not pre_retrieved_precedents:
            self._log("⚠️ No knowledge base provided", "warning")
            return ReasonerOutput(
                claim=claim,
                causality_classification={
                    "domain": routing_decision.domain,
                    "causal_type_id": "",
                    "theory_id": "",
                    "source": "router",
                    "reason": "empty_kb",
                },
                causal_type_id="",
                theory_id="",
                anchor_norms={},
                principle_tests=[],
                relevant_statutes=[],
                relevant_precedents=[],
                arguments=[],
                reasoning_chain=[
                    "No statutes or precedents were provided for analysis."
                ],
                raw_response="Analysis not completed: no statutory or case sources available.",
            )

        # ── Causality-dependent phases ──────────────────────────────────
        domain = routing_decision.domain
        self._log(f"🔬 Router domain: {domain}")

        hint_causal_id = routing_decision.causal_type_id or ""
        hint_theory_id = routing_decision.theory_id or ""

        if enable_causality:
            self._log(
                "🔬 Causality ENABLED — classification runs post-chain and is passed to Counter"
            )
            chain_class = {
                "domain": domain,
                "causal_type_id": "",
                "theory_id": "",
                "source": "pending_posthoc",
            }
        else:
            self._log(
                "🔬 Causality DISABLED — no causal classification/attack taxonomy"
            )
            chain_class = {
                "domain": domain,
                "causal_type_id": "",
                "theory_id": "",
                "source": "disabled",
            }

        # Post-hoc values are populated after the final reasoning chain is generated.
        final_causal_id = ""
        final_theory_id = ""
        causal_types_for_counter: list[str] = []
        anchor_norms: dict = {}
        anchor_statutes: list[dict] = []
        principle_tests: list[dict] = []

        # Phase 3: refine reasoning with anchor norms + cross-ref expansion
        self._log(
            f"📌 Phase 3: merging pre_retrieved ({len(pre_retrieved_statutes)}) + anchor ({len(anchor_statutes)}) statutes"
        )
        all_statutes = pre_retrieved_statutes + anchor_statutes
        statute_origin_map: dict[tuple[str, str], set[str]] = {}
        for s in pre_retrieved_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add("pre_retrieval")
        for s in anchor_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add(
                str(s.get("_kb_origin") or "taxonomy_anchor")
            )
        seen_keys = set()
        deduped_statutes = []
        for s in all_statutes:
            key = (s.get("articolo"), s.get("source"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_statutes.append(s)
        self._log(f"📌 After dedup: {len(deduped_statutes)} statutes")
        before_expand = len(deduped_statutes)
        before_expand_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        deduped_statutes = self._expand_with_cross_references(deduped_statutes)
        after_expand_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        for k in after_expand_keys - before_expand_keys:
            statute_origin_map.setdefault(k, set()).add("cross_ref")
        if len(deduped_statutes) > before_expand:
            self._log(
                f"➕ Added {len(deduped_statutes) - before_expand} statutes via cross-ref",
                "info",
            )
        self._log_final_statute_origins(
            deduped_statutes,
            statute_origin_map,
            label="Reasoner KB",
        )

        allowed_statutes = [
            f"Art. {s.get('articolo')} ({self._source_short_label(s.get('source', ''))})"
            for s in deduped_statutes
        ]
        allowed_precedents = [
            p.get("title", "Untitled") for p in pre_retrieved_precedents
        ]
        knowledge_base = self._format_context_for_prompt(
            deduped_statutes, pre_retrieved_precedents
        )
        anchor_text = self._format_anchor_norms(anchor_norms)
        principle_text = self._format_principle_tests(principle_tests)

        # DEBUG: log what's being injected
        self._log(
            f"📝 Anchor text for prompt:\n{anchor_text[:500] if len(anchor_text) > 500 else anchor_text}"
        )
        self._log(
            f"📝 Principle tests for prompt:\n{principle_text[:500] if len(principle_text) > 500 else principle_text}"
        )

        # ----------------------------------------------------------
        # Phase 3: iterative step-by-step chain generation
        # ----------------------------------------------------------
        MAX_CHAIN_RETRIES = settings.chain_max_retries
        output = None

        for attempt in range(1, MAX_CHAIN_RETRIES + 1):
            self._log(f"🔄 Reasoner generation attempt {attempt}/{MAX_CHAIN_RETRIES}")
            did_posthoc_anchor_refine = False

            try:
                raw_output, iterative_chain = self._generate_chain_iteratively(
                    claim=claim,
                    routing_decision=routing_decision,
                    anchor_text=anchor_text,
                    principle_text=principle_text,
                    knowledge_base=knowledge_base,
                    allowed_statutes=allowed_statutes,
                    allowed_precedents=allowed_precedents,
                    stream_callback=stream_callback,
                )
            except Exception as gen_exc:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: planner/executor failed ({gen_exc})",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    raise
                continue

            # Generate dynamic LLM conclusion from chain
            conclusion = (
                self._generate_conclusion(
                    claim, iterative_chain, stream_callback=stream_callback
                )
                if iterative_chain
                else ""
            )
            if conclusion:
                raw_output = self._assemble_raw_response(
                    claim, iterative_chain, conclusion_text=conclusion
                )

            # Use iterative chain directly; fall back to extraction if empty
            reasoning_chain = (
                iterative_chain
                if iterative_chain
                else self._extract_reasoning_chain(raw_output)
            )
            reasoning_chain = self._sanitize_reasoning_chain(
                reasoning_chain, pre_retrieved_precedents
            )
            arguments = self._extract_arguments(raw_output)

            causality_classification = {
                **chain_class,
                "domain": domain,
                "source": chain_class.get("source", "pending_posthoc"),
            }
            mismatch_status = ""
            final_causal_id = ""
            final_theory_id = ""
            causal_types_for_counter = []
            anchor_norms = {}
            principle_tests = []

            # Post-hoc causal classification on the FINAL chain.
            if enable_causality and reasoning_chain:
                try:
                    post_chain_class = self._classify_causality_from_reasoning(
                        claim=claim,
                        reasoning_chain=reasoning_chain,
                        raw_response=raw_output,
                        domain=domain,
                    )
                    post_causal_id = post_chain_class.get("causal_type_id") or ""
                    post_theory_id = post_chain_class.get("theory_id") or ""
                    if post_causal_id:
                        post_causal_id, post_theory_id = config_loader.validate_ids(
                            post_causal_id, post_theory_id
                        )
                    final_causal_id = post_causal_id
                    final_theory_id = post_theory_id or ""
                    causality_classification = {
                        **post_chain_class,
                        "domain": domain,
                        "source": "reasoning_chain_posthoc",
                    }

                    if final_causal_id:
                        causal_types_for_counter = (
                            self._build_causal_type_bundle_for_counter(
                                claim=claim,
                                domain=domain,
                                primary_causal_type_id=final_causal_id,
                                reasoning_chain=reasoning_chain,
                                raw_response=raw_output,
                            )
                        )
                        anchor_norms, anchor_statutes_posthoc, principle_tests = (
                            self._filtered_anchor_norms_for_types(
                                causal_types_for_counter, claim
                            )
                        )
                        self._log(
                            f"📋 Post-hoc anchor norms: core={len(anchor_norms.get('core_norms', []))}, accessory={len(anchor_norms.get('accessory_norms', []))}"
                        )
                        self._log(
                            "🔬 Counter causal-type bundle: "
                            f"{causal_types_for_counter}",
                            "info",
                        )

                        if not did_posthoc_anchor_refine:
                            refinement = (
                                self._maybe_refine_reasoning_with_posthoc_anchors(
                                    claim=claim,
                                    routing_decision=routing_decision,
                                    pre_retrieved_precedents=pre_retrieved_precedents,
                                    deduped_statutes=deduped_statutes,
                                    statute_origin_map=statute_origin_map,
                                    anchor_norms=anchor_norms,
                                    anchor_statutes=anchor_statutes_posthoc,
                                    principle_tests=principle_tests,
                                    reasoning_chain=reasoning_chain,
                                    raw_output=raw_output,
                                )
                            )
                            if refinement:
                                did_posthoc_anchor_refine = True
                                raw_output = refinement["raw_output"]
                                reasoning_chain = refinement["reasoning_chain"]
                                conclusion = refinement["conclusion"] or conclusion
                                arguments = refinement["arguments"]
                                deduped_statutes = refinement["deduped_statutes"]
                                statute_origin_map = refinement["statute_origin_map"]

                                # Re-run post-hoc classification/bundle on refined chain once.
                                post_chain_class = (
                                    self._classify_causality_from_reasoning(
                                        claim=claim,
                                        reasoning_chain=reasoning_chain,
                                        raw_response=raw_output,
                                        domain=domain,
                                    )
                                )
                                post_causal_id = (
                                    post_chain_class.get("causal_type_id") or ""
                                )
                                post_theory_id = post_chain_class.get("theory_id") or ""
                                if post_causal_id:
                                    post_causal_id, post_theory_id = (
                                        config_loader.validate_ids(
                                            post_causal_id, post_theory_id
                                        )
                                    )
                                final_causal_id = post_causal_id
                                final_theory_id = post_theory_id or ""
                                causality_classification = {
                                    **post_chain_class,
                                    "domain": domain,
                                    "source": "reasoning_chain_posthoc_refined",
                                }
                                if final_causal_id:
                                    causal_types_for_counter = (
                                        self._build_causal_type_bundle_for_counter(
                                            claim=claim,
                                            domain=domain,
                                            primary_causal_type_id=final_causal_id,
                                            reasoning_chain=reasoning_chain,
                                            raw_response=raw_output,
                                        )
                                    )
                                    (
                                        anchor_norms,
                                        _anchor_statutes_posthoc_refined,
                                        principle_tests,
                                    ) = self._filtered_anchor_norms_for_types(
                                        causal_types_for_counter, claim
                                    )
                                    self._log(
                                        "📋 Post-hoc anchor norms (after refinement): "
                                        f"core={len(anchor_norms.get('core_norms', []))}, "
                                        f"accessory={len(anchor_norms.get('accessory_norms', []))}"
                                    )
                                    self._log(
                                        "🔬 Counter causal-type bundle (after refinement): "
                                        f"{causal_types_for_counter}",
                                        "info",
                                    )

                    # Optional diagnostics: compare eventual input hint vs post-hoc chain class.
                    if hint_causal_id:
                        causality_classification["input_hint_causal_type_id"] = (
                            hint_causal_id
                        )
                        causality_classification["input_hint_theory_id"] = (
                            hint_theory_id or ""
                        )
                        if final_causal_id and hint_causal_id != final_causal_id:
                            mismatch_status = "hint_chain_drift"
                            self._log(
                                "⚠️ Post-hoc chain causality differs from input hint: "
                                f"hint={hint_causal_id} vs chain={final_causal_id}",
                                "warning",
                            )
                except Exception as class_exc:
                    self._log(
                        f"⚠️ Post-hoc causality classification failed: {class_exc}",
                        "warning",
                    )
            elif enable_causality:
                mismatch_status = "posthoc_unavailable_no_chain"

            output = ReasonerOutput(
                claim=claim,
                causality_classification=causality_classification,
                causal_type_id=final_causal_id,
                theory_id=final_theory_id or "",
                causal_type_ids_for_counter=causal_types_for_counter,
                mismatch_status=mismatch_status,
                anchor_norms=anchor_norms,
                principle_tests=principle_tests,
                relevant_statutes=deduped_statutes,
                relevant_precedents=pre_retrieved_precedents,
                arguments=arguments,
                reasoning_chain=reasoning_chain,
                raw_response=raw_output,
                conclusion=conclusion,
            )

            formatter = AspicFormatter(
                role="support",
                statutes=output.relevant_statutes,
                precedents=output.relevant_precedents,
            )
            output.aspic_ir = formatter.format(
                claim=claim,
                raw_response=output.raw_response,
                reasoning_chain=output.reasoning_chain,
                arguments=output.arguments,
                metadata={
                    "causal_type_id": final_causal_id,
                    "theory_id": final_theory_id,
                },
            )

            # Validate: ASPIC_IR must contain reasoning chain nodes (S1, S2, …)
            if self._has_valid_reasoning_chain(output.aspic_ir):
                break
            else:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: empty reasoning chain "
                    f"(no S* nodes in ASPIC_IR) — retrying…",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    self._log(
                        f"❌ Failed to generate a valid reasoning chain after "
                        f"{MAX_CHAIN_RETRIES} attempts",
                        "error",
                    )

        assert output is not None, "ReasonerOutput was never assigned"  # guard for mypy
        chain_len = (
            len(output.aspic_ir.get("reasoning_chain", [])) if output.aspic_ir else 0
        )
        self._log(
            f"✅ Generated {len(output.arguments)} argument(s), "
            f"{chain_len} reasoning steps",
            "success",
        )
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke_reasoner(self, prompt: str) -> tuple[str, list]:
        """Invoke the ReAct agent with resilient retry/key-rotation/fallback."""
        messages = [HumanMessage(content=prompt)]
        try:
            result = resilient_react_invoke(
                self._build_react_agent,
                {"messages": messages},
                model_order=self._resilient_model_order(),
            )
        except Exception as e:
            # Handle Groq tool_use_failed: the model generated a valid response
            # but the tool-calling mechanism failed. Extract the response.
            raw_response = self._extract_failed_generation(e)
            if raw_response:
                self._log(
                    "⚠️ Tool call failed but valid response recovered from failed_generation",
                    "warning",
                )
                return raw_response, []
            raise

        messages_out = result.get("messages", [])

        tool_names: list[str] = []
        for msg in messages_out:
            if hasattr(msg, "name") and msg.name:
                tool_names.append(msg.name)
        if tool_names:
            self._log(f"📊 Tools used: {', '.join(set(tool_names))}")

        raw_output = ""
        for msg in reversed(messages_out):
            if isinstance(msg, ToolMessage):
                continue
            msg_content = getattr(msg, "content", None)
            if msg_content:
                raw_output = str(msg_content)
                break
        return raw_output, messages_out

    def _extract_failed_generation(self, exc: Exception) -> str:
        """Extract the valid response from a Groq tool_use_failed error."""
        error_str = str(exc)
        if "tool_use_failed" not in error_str:
            return ""
        # Try to extract from exception body (groq.BadRequestError)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_data = body.get("error", {})
            failed = error_data.get("failed_generation", "")
            if failed and len(failed) > 50:
                return failed
        # Fallback: parse from string representation
        marker = "'failed_generation': \""
        idx = error_str.find(marker)
        if idx == -1:
            marker = "'failed_generation': '"
            idx = error_str.find(marker)
        if idx != -1:
            start = idx + len(marker)
            # Find the closing quote
            end = error_str.find('"}}', start)
            if end == -1:
                end = error_str.find("'}", start)
            if end != -1:
                text = error_str[start:end]
                # Unescape newlines
                text = text.replace("\\n", "\n")
                if len(text) > 50:
                    return text
        return ""

    def _extract_cited_articles(self, text: str) -> list[str]:
        """Extract article references cited in the reasoning chain."""
        mentions = extract_article_mentions(text, require_code=True)
        articles = [
            format_article_citation(m.article_id, m.source_hint) for m in mentions
        ]
        # Deduplica mantenendo ordine
        seen = set()
        unique = []
        for a in articles:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    @staticmethod
    def _article_ids_by_source(
        mentions: list,
    ) -> tuple[set[str], set[str], set[str]]:
        """Group cited article ids by source hint (penale/civile/amministrativo)."""
        pen: set[str] = set()
        civ: set[str] = set()
        amm: set[str] = set()
        for mention in mentions:
            article_id = normalize_article_id(getattr(mention, "article_id", ""))
            source_hint = str(getattr(mention, "source_hint", "") or "").strip().lower()
            if not article_id:
                continue
            if source_hint == "codice_penale":
                pen.add(article_id)
            elif source_hint == "codice_civile":
                civ.add(article_id)
            elif source_hint == "codice_amministrativo":
                amm.add(article_id)
        return pen, civ, amm

    def _heuristic_causal_type_from_mentions(
        self,
        *,
        mentions: list,
        allowed_ids: list[str],
    ) -> str:
        """
        Deterministic fallback classifier based on cited-article signatures.

        This reduces over-fallback to the first allowed id and activates
        finer causal types introduced in taxonomy.
        """
        allowed = set(allowed_ids)
        pen, civ, amm = self._article_ids_by_source(mentions)

        def can(ct_id: str) -> bool:
            return ct_id in allowed

        # Mixed civil+criminal profile
        if pen and civ and can("MIXED_PEN_CIV_CONCURRENCE"):
            return "MIXED_PEN_CIV_CONCURRENCE"

        # Penal signatures
        if can("PEN_RIGHTS_BALANCING") and pen.intersection(
            {"51", "595", "610", "392", "393", "614"}
        ):
            return "PEN_RIGHTS_BALANCING"
        if can("PEN_PROPERTY_QUALIFICATION") and pen.intersection(
            {"646", "640", "624", "624-bis", "625"}
        ):
            return "PEN_PROPERTY_QUALIFICATION"
        if can("PEN_COLPA_GRADATION") and pen.intersection(
            {"42", "43", "61", "62-bis", "113", "133"}
        ):
            return "PEN_COLPA_GRADATION"
        if can("PEN_INTERVENING") and pen.intersection({"41", "45"}):
            return "PEN_INTERVENING"
        if can("IMPUTATION_FILTER") and "45" in pen:
            return "IMPUTATION_FILTER"
        if can("PEN_FACTUAL") and pen.intersection({"40", "41", "589-bis", "590"}):
            return "PEN_FACTUAL"

        # Civil signatures
        if can("CIV_SALE_DEFECTS") and civ.intersection(
            {"1490", "1491", "1492", "1494", "1495"}
        ):
            return "CIV_SALE_DEFECTS"
        if can("CIV_CONTRACT_REMEDIES") and civ.intersection(
            {"1218", "1453", "1455", "1457", "1460", "1385", "1382", "1384"}
        ):
            return "CIV_CONTRACT_REMEDIES"
        if can("CIV_CUSTODY_DAMAGE") and civ.intersection(
            {"2051", "2052", "1117", "1123", "1126"}
        ):
            return "CIV_CUSTODY_DAMAGE"
        if can("CIV_REMOTENESS") and civ.intersection(
            {"1223", "1225", "1226", "1227", "2056"}
        ):
            return "CIV_REMOTENESS"

        # Administrative signatures
        if can("AMM_AUTOTUTELA_BALANCE") and amm.intersection(
            {"21-novies", "21-quinquies"}
        ):
            return "AMM_AUTOTUTELA_BALANCE"
        if can("AMM_DELAY_REMEDIES") and amm.intersection({"2-bis"}):
            return "AMM_DELAY_REMEDIES"
        if can("AMM_PROCEDURAL_LEGITIMACY") and amm:
            return "AMM_PROCEDURAL_LEGITIMACY"

        return ""

    def _build_causal_type_bundle_for_counter(
        self,
        *,
        claim: str,
        domain: str,
        primary_causal_type_id: str,
        reasoning_chain: list[str],
        raw_response: str,
        max_total: int | None = None,
    ) -> list[str]:
        """Build a small multi-type bundle for counter anchor norms.

        Keeps the post-hoc classified type as primary, then adds a small number of
        complementary causal types using article signatures and generic claim cues.
        This is intentionally conservative and mainly targets mixed penal
        causation/culpability patterns without changing the primary routing.
        """
        primary = (primary_causal_type_id or "").strip()
        if not primary:
            return []

        config = config_loader.load_config()
        ct_index = config_loader.causal_types_by_id(config)
        domain_lower = (domain or "").strip().lower()
        if domain_lower == "entrambi":
            allowed_ids = list(ct_index.keys())
        else:
            allowed_ids = [
                ct_id
                for ct_id, ct in ct_index.items()
                if str(ct.get("domain", "") or "").strip().lower() == domain_lower
            ] or list(ct_index.keys())

        allowed = set(allowed_ids)
        effective_max_total = (
            max_total
            if max_total is not None
            else (4 if domain_lower == "penale" else 3)
        )
        selected: list[str] = []
        reasons: list[str] = []
        truncated: list[str] = []

        def add(ct_id: str, reason: str) -> None:
            if not ct_id or ct_id not in allowed or ct_id in selected:
                return
            if len(selected) >= effective_max_total:
                truncated.append(f"{ct_id}: {reason}")
                return
            selected.append(ct_id)
            reasons.append(f"{ct_id}: {reason}")

        add(primary, "primary post-hoc classification")

        chain_text = "\n".join(reasoning_chain) or raw_response or ""
        mentions = extract_article_mentions(chain_text, require_code=True)
        pen, _civ, _amm = self._article_ids_by_source(mentions)

        claim_text = (claim or "").lower()
        chain_lower = chain_text.lower()
        penal_text = f"{claim_text}\n{chain_lower}"

        # Article-signature complements (deterministic)
        if pen.intersection({"40", "41", "589", "589-bis", "590"}):
            add("PEN_FACTUAL", "penal causal/factual article signature")
        if pen.intersection({"41", "45", "54", "90"}):
            add("PEN_INTERVENING", "intervening/imputation article signature")
        if pen.intersection({"45", "54", "90"}):
            add(
                "IMPUTATION_FILTER", "fortuito/necessity/imputability article signature"
            )
        if pen.intersection({"42", "43", "61", "62-bis", "113", "133"}):
            add("PEN_COLPA_GRADATION", "fault gradation/sanction article signature")

        # Generic penal claim cues (used when cited articles are incomplete)
        has_counterfactual_cues = any(
            cue in penal_text
            for cue in (
                "nesso causale",
                "controfatt",
                "causa sopravven",
                "si sarebbe verificat",
                "perizi",
            )
        )
        has_emergency_or_emotion_cues = any(
            cue in penal_text
            for cue in (
                "stato emotiv",
                "urgenza",
                "parto",
                "ospedal",
                "fortuit",
                "forza maggiore",
                "stato di necessit",
                "necessità",
            )
        )
        has_fault_gradation_cues = any(
            cue in penal_text
            for cue in (
                "colpa",
                "prevedibil",
                "evitabil",
                "attenuan",
                "aggravan",
                "rimprover",
                "commisur",
            )
        )

        if has_counterfactual_cues:
            add("PEN_FACTUAL", "counterfactual/causal cues in claim or chain")
            add(
                "PEN_INTERVENING",
                "supervening-cause style causal cues in claim or chain",
            )
        if has_emergency_or_emotion_cues:
            add("PEN_INTERVENING", "emergency/emotional-state cues in claim or chain")
            add(
                "IMPUTATION_FILTER",
                "emergency/fortuitous/imputability cues in claim or chain",
            )
        if has_fault_gradation_cues:
            add(
                "PEN_COLPA_GRADATION", "fault gradation/sanction cues in claim or chain"
            )

        if reasons:
            self._log(
                "🔬 Counter causal-type bundle reasons: " + " | ".join(reasons),
                "info",
            )
        if truncated:
            self._log(
                "⚠️ Counter causal-type bundle truncated "
                f"(max_total={effective_max_total}): " + " | ".join(truncated),
                "warning",
            )

        return selected

    def _classify_causality_from_reasoning(
        self, claim: str, reasoning_chain: list[str], raw_response: str, domain: str
    ) -> dict:
        """Classify causality based on the articles cited in the reasoning chain, filtered by domain."""
        config = config_loader.load_config()
        ct_index = config_loader.causal_types_by_id(config)

        # Filter causal types by domain
        domain_lower = domain.lower()
        if domain_lower == "entrambi":
            allowed_ids = list(ct_index.keys())
        else:
            # Specific domain selected by router (es. civile, penale, amministrativo)
            allowed_ids = [
                ct_id
                for ct_id, ct in ct_index.items()
                if ct.get("domain", "").lower() == domain_lower
            ]

        if not allowed_ids:
            # Fallback to all if no match
            allowed_ids = list(ct_index.keys())

        self._log(f"🔬 Allowed causal_type_ids for domain '{domain}': {allowed_ids}")

        chain_text = "\n".join(reasoning_chain) or raw_response or ""
        mentions = extract_article_mentions(chain_text, require_code=True)

        # Estrai gli articoli citati dalla catena di ragionamento
        cited_articles = self._extract_cited_articles(chain_text)
        articles_text = (
            ", ".join(cited_articles) if cited_articles else "Nessun articolo citato"
        )

        self._log(f"🔬 Articoli citati nella catena di ragionamento: {articles_text}")

        heuristic_id = self._heuristic_causal_type_from_mentions(
            mentions=mentions,
            allowed_ids=allowed_ids,
        )
        if heuristic_id:
            pen_ids, _civ_ids, _amm_ids = self._article_ids_by_source(mentions)
            factual_intervening_markers = pen_ids.intersection(
                {"40", "41", "45", "589-bis", "590"}
            )
            if heuristic_id in {"PEN_FACTUAL", "PEN_INTERVENING"} and (
                factual_intervening_markers == {"41"}
            ):
                self._log(
                    "⚠️ Heuristic causality classification ambiguous on shared marker "
                    "Art. 41 c.p.; deferring to LLM classifier",
                    "warning",
                )
            else:
                theory_id = self._get_default_theory(heuristic_id, config)
                result = {"causal_type_id": heuristic_id, "theory_id": theory_id}
                self._log(f"🔬 Heuristic causality classification: {result}")
                return result

        # Build causal type descriptions for prompt
        type_descriptions = []
        for ct_id in allowed_ids:
            ct = ct_index.get(ct_id, {})
            type_descriptions.append(
                f"- {ct_id}: {ct.get('name', '')} [{ct.get('domain', '')}]"
            )

        prompt = render_prompt(
            "reasoner.classify_causality",
            domain=domain,
            type_descriptions=chr(10).join(type_descriptions),
            claim=claim,
            articles_text=articles_text,
            chain_text=chain_text,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            content = (resp.content or "").strip()

            # Parse causal_type_id from LLM response
            causal_type_id = self._extract_causal_type_id(content, allowed_ids)

            if causal_type_id:
                # Get theory_id from default_mapping
                theory_id = self._get_default_theory(causal_type_id, config)
                result = {"causal_type_id": causal_type_id, "theory_id": theory_id}
                self._log(f"🔬 Classificazione basata su articoli: {result}")
                return result
            else:
                # Fallback to first allowed
                self._log(
                    f"⚠️ Could not parse causal_type_id from '{content}', using first allowed: {allowed_ids[0]}",
                    "warning",
                )
                default_theory = self._get_default_theory(allowed_ids[0], config)
                return {"causal_type_id": allowed_ids[0], "theory_id": default_theory}
        except Exception as e:
            self._log(f"⚠️ Causality classification on reasoning failed: {e}", "warning")

        # Fallback to first allowed with default theory
        fallback_causal = allowed_ids[0] if allowed_ids else ""
        fallback_theory = (
            self._get_default_theory(fallback_causal, config) if fallback_causal else ""
        )
        return {"causal_type_id": fallback_causal, "theory_id": fallback_theory}

    def _get_default_theory(self, causal_type_id: str, config: dict) -> str:
        """Get the default theory_id for a causal_type_id from config."""
        default_mapping = config.get("default_mapping", [])
        for mapping in default_mapping:
            if mapping.get("causal_type") == causal_type_id:
                return mapping.get("reasoner_primary_theory", "")
        return ""

    def _extract_causal_type_id(self, response: str, allowed_ids: list[str]) -> str:
        """Extract causal_type_id from LLM response by matching against allowed ids."""
        response_clean = (
            response.strip().replace("`", "").replace('"', "").replace("'", "")
        )
        # Try exact match first
        for ct_id in allowed_ids:
            if ct_id == response_clean:
                return ct_id
        # Try substring match
        for ct_id in allowed_ids:
            if ct_id in response_clean:
                return ct_id
        # Try case-insensitive match
        for ct_id in allowed_ids:
            if ct_id.lower() in response_clean.lower():
                return ct_id
        return ""

    def _parse_json_like(self, text: str) -> dict:
        """Parse LLM JSON-ish content."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return {
                    "causal_type_id": str(data.get("causal_type_id", "")).strip(),
                    "theory_id": str(data.get("theory_id", "")).strip(),
                }
        except Exception:
            pass
        return {}

    def _filtered_anchor_norms_for_types(
        self, causal_types: list[str], claim: str
    ) -> tuple[dict, list[dict], list[dict]]:
        """
        Retrieve and filter anchor norms (core/accessory) by claim for the given causal types.
        Returns (merged_anchor_norms, statutes_from_norms, merged_principle_tests).
        """
        unique_cts = list(dict.fromkeys(ct for ct in causal_types if ct))
        merged_core: list[dict] = []
        merged_accessory: list[dict] = []
        principle_tests: list[dict] = []
        statutes: list[dict] = []

        for ct in unique_cts:
            try:
                theory = get_causality_theory_tool.invoke(
                    {"causality_type": ct, "claim": claim}
                )
            except Exception as e:
                self._log(f"⚠️ Failed to load theory for {ct}: {e}", "warning")
                continue

            core_rel = theory.get("norme_core_rilevanti", []) or []
            acc_rel = theory.get("norme_accessorie_rilevanti", []) or []
            core_full = theory.get("norme_core", []) or []
            acc_full = theory.get("norme_accessorie", []) or []

            # If no filtered norms available, use full norms (happens when no claim filter applied)
            if not core_rel and core_full:
                core_rel = core_full
            if not acc_rel and acc_full:
                acc_rel = acc_full

            taxonomy_norms = core_rel + acc_rel

            # Support both old keys (riferimento) and new keys (ref)
            kept_refs = [
                n.get("ref") or n.get("riferimento")
                for n in taxonomy_norms
                if n.get("ref") or n.get("riferimento")
            ]
            kept_set = {r for r in kept_refs if r}
            discarded_refs = [
                n.get("ref") or n.get("riferimento")
                for n in (core_full + acc_full)
                if (n.get("ref") or n.get("riferimento"))
                and (n.get("ref") or n.get("riferimento")) not in kept_set
            ]
            self._log(
                f"🔎 [taxonomy] Causality {ct}: core {len(core_rel)}/{len(core_full)}, accessory {len(acc_rel)}/{len(acc_full)}"
            )
            if kept_refs:
                self._log(f"   ✔️ Kept: {', '.join(kept_refs)}")
            if discarded_refs:
                self._log(f"   ❌ Discarded: {', '.join(discarded_refs)}")

            # Second-stage applicability on taxonomy-derived statutes:
            # - core norms: soft (flag but keep even if rejected)
            # - accessory norms: hard (drop if rejected)
            core_pairs = [
                (n, self._norm_to_statute_dict(n))
                for n in core_rel
                if n.get("ref") or n.get("riferimento")
            ]
            acc_pairs = [
                (n, self._norm_to_statute_dict(n))
                for n in acc_rel
                if n.get("ref") or n.get("riferimento")
            ]
            core_statutes_raw = [st for _, st in core_pairs]
            acc_statutes_raw = [st for _, st in acc_pairs]
            (
                _core_applicable,
                acc_applicable,
                _core_rejected,
                _acc_rejected,
            ) = self._filter_taxonomy_anchor_statutes_by_applicability(
                claim,
                core_statutes=core_statutes_raw,
                accessory_statutes=acc_statutes_raw,
                log_prefix=f"reasoner/taxonomy/{ct}",
            )
            acc_keep_keys = {self._statute_identity_key(st) for st in acc_applicable}
            acc_rel = [
                norm
                for norm, st in acc_pairs
                if self._statute_identity_key(st) in acc_keep_keys
            ]
            taxonomy_norms = core_rel + acc_rel

            merged_core.extend(core_rel)
            merged_accessory.extend(acc_rel)
            statutes.extend(
                [
                    self._norm_to_statute_dict(n)
                    for n in taxonomy_norms
                    if n.get("ref") or n.get("riferimento")
                ]
            )
            pt = theory.get("principio_test") or theory.get("principle_tests") or []
            if isinstance(pt, list):
                principle_tests.extend(pt)

        anchor_norms = {
            "core_norms": merged_core,
            "accessory_norms": merged_accessory,
        }
        return anchor_norms, statutes, principle_tests

    def _maybe_refine_reasoning_with_posthoc_anchors(
        self,
        *,
        claim: str,
        routing_decision: RoutingDecision,
        pre_retrieved_precedents: list[dict],
        deduped_statutes: list[dict],
        statute_origin_map: dict[tuple[str, str], set[str]],
        anchor_norms: dict,
        anchor_statutes: list[dict],
        principle_tests: list[dict],
        reasoning_chain: list[str],
        raw_output: str,
    ) -> dict | None:
        """One-shot refinement when post-hoc core anchors are missing from the Reasoner KB."""
        if not anchor_statutes:
            return None

        kb_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        kb_article_ids = {
            normalize_article_id(str(s.get("articolo", "") or ""))
            for s in deduped_statutes
            if s.get("articolo")
        }

        core_anchor_ids: set[str] = set()
        for norm in anchor_norms.get("core_norms", []) or []:
            ref = str(norm.get("ref") or norm.get("riferimento") or "").strip()
            if ref:
                core_anchor_ids.add(normalize_article_id(ref))
        core_anchor_ids.discard("")

        missing_core_ids = sorted(
            aid for aid in core_anchor_ids if aid not in kb_article_ids
        )
        if not missing_core_ids:
            return None

        extra_anchor_statutes = []
        for st in anchor_statutes:
            key = self._statute_identity_key(st)
            if key in kb_keys:
                continue
            payload = dict(st)
            payload["_kb_origin"] = str(
                payload.get("_kb_origin") or "taxonomy_posthoc_anchor"
            )
            extra_anchor_statutes.append(payload)

        if not extra_anchor_statutes:
            return None

        self._log(
            "🛠️ Reasoner post-hoc anchor refinement triggered: missing core anchor(s) in KB "
            f"{', '.join(f'Art. {aid}' for aid in missing_core_ids)}; "
            f"injecting {len(extra_anchor_statutes)} anchor statute(s) and regenerating once",
            "warning",
        )

        refined_origin_map: dict[tuple[str, str], set[str]] = {
            k: set(v) for k, v in statute_origin_map.items()
        }
        for s in extra_anchor_statutes:
            k = self._statute_identity_key(s)
            refined_origin_map.setdefault(k, set()).add(
                str(s.get("_kb_origin") or "taxonomy_posthoc_anchor")
            )

        seen_keys = set()
        refined_statutes = []
        for s in deduped_statutes + extra_anchor_statutes:
            key = self._statute_identity_key(s)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            refined_statutes.append(s)

        before_expand = len(refined_statutes)
        before_expand_keys = {self._statute_identity_key(s) for s in refined_statutes}
        refined_statutes = self._expand_with_cross_references(refined_statutes)
        after_expand_keys = {self._statute_identity_key(s) for s in refined_statutes}
        for k in after_expand_keys - before_expand_keys:
            refined_origin_map.setdefault(k, set()).add("cross_ref")
        if len(refined_statutes) > before_expand:
            self._log(
                f"➕ Added {len(refined_statutes) - before_expand} statutes via cross-ref (refinement)",
                "info",
            )
        self._log_final_statute_origins(
            refined_statutes,
            refined_origin_map,
            label="Reasoner KB (refined)",
        )

        allowed_statutes = [
            f"Art. {s.get('articolo')} ({self._source_short_label(s.get('source', ''))})"
            for s in refined_statutes
        ]
        allowed_precedents = [
            p.get("title", "Untitled") for p in pre_retrieved_precedents
        ]
        knowledge_base = self._format_context_for_prompt(
            refined_statutes, pre_retrieved_precedents
        )
        anchor_text = self._format_anchor_norms(anchor_norms)
        principle_text = self._format_principle_tests(principle_tests)

        try:
            raw_output_refined, iterative_chain = self._generate_chain_iteratively(
                claim=claim,
                routing_decision=routing_decision,
                anchor_text=anchor_text,
                principle_text=principle_text,
                knowledge_base=knowledge_base,
                allowed_statutes=allowed_statutes,
                allowed_precedents=allowed_precedents,
                stream_callback=None,
            )
        except Exception as exc:
            self._log(
                f"⚠️ Reasoner post-hoc anchor refinement failed during regeneration: {exc}",
                "warning",
            )
            return None

        conclusion_refined = (
            self._generate_conclusion(claim, iterative_chain, stream_callback=None)
            if iterative_chain
            else ""
        )
        if conclusion_refined:
            raw_output_refined = self._assemble_raw_response(
                claim, iterative_chain, conclusion_text=conclusion_refined
            )

        reasoning_chain_refined = (
            iterative_chain
            if iterative_chain
            else self._extract_reasoning_chain(raw_output_refined)
        )
        reasoning_chain_refined = self._sanitize_reasoning_chain(
            reasoning_chain_refined, pre_retrieved_precedents
        )
        if not reasoning_chain_refined:
            self._log(
                "⚠️ Reasoner post-hoc anchor refinement produced empty reasoning chain; keeping original",
                "warning",
            )
            return None

        arguments_refined = self._extract_arguments(raw_output_refined)
        return {
            "raw_output": raw_output_refined,
            "reasoning_chain": reasoning_chain_refined,
            "conclusion": conclusion_refined,
            "arguments": arguments_refined,
            "deduped_statutes": refined_statutes,
            "statute_origin_map": refined_origin_map,
        }

    def _generate_chain_iteratively(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        anchor_text: str,
        principle_text: str,
        knowledge_base: str,
        allowed_statutes: list[str],
        allowed_precedents: list[str],
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> tuple[str, list[str]]:
        """Generate the reasoning chain with plan -> execute -> residual replan workflow."""
        max_steps = settings.chain_max_steps
        min_steps = settings.chain_min_steps
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )
        steps: list[str] = []
        step_summaries: list[str] = []
        used_norms: list[str] = []
        plan_round = 0
        stalled_rounds = 0
        max_plan_rounds = max(1, self._max_plan_retries + 1)

        while len(steps) < min_steps and len(steps) < max_steps:
            plan_round += 1
            if plan_round > max_plan_rounds:
                break

            remaining_min = max(1, min_steps - len(steps))
            remaining_max = max(1, max_steps - len(steps))
            planner_mode = "RESUME" if steps else "FULL"

            plan = self._generate_reasoning_plan(
                claim=claim,
                routing_decision=routing_decision,
                anchor_text=anchor_text,
                principle_text=principle_text,
                knowledge_base=knowledge_base,
                statutes_list=statutes_list,
                precedents_list=precedents_list,
                min_steps=remaining_min,
                max_steps=remaining_max,
                planner_mode=planner_mode,
                resume_from_step=len(steps) + 1,
                existing_summaries=step_summaries,
            )
            plan = self._coerce_plan_to_allowed_norms(
                plan=plan,
                allowed_statutes=allowed_statutes,
            )
            plan = self._prune_plan_against_existing_history(
                plan=plan,
                previous_summaries=step_summaries,
            )

            if not plan:
                stalled_rounds += 1
                self._log(
                    "⚠️ Planner produced only redundant residual steps; retrying residual plan",
                    "warning",
                )
                if stalled_rounds >= 2:
                    break
                continue

            self._log(
                f"🧭 Reasoning plan generated: {len(plan)} step(s) "
                f"[round={plan_round}, mode={planner_mode}, completed={len(steps)}]"
            )

            steps_before_round = len(steps)
            round_failed = False

            for local_idx, plan_step in enumerate(plan, start=1):
                global_idx = len(steps) + 1
                self._log(
                    f"🔗 Generating planned step {global_idx}/{max_steps}: "
                    f"{plan_step.get('goal', '')[:80]}"
                )
                step_text = self._generate_support_step_from_plan(
                    claim=claim,
                    routing_decision=routing_decision,
                    anchor_text=anchor_text,
                    principle_text=principle_text,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    precedents_list=precedents_list,
                    plan=plan,
                    plan_index=local_idx,
                    plan_step=plan_step,
                    previous_steps=steps,
                    previous_summaries=step_summaries,
                    used_norms=used_norms,
                    stream_callback=stream_callback,
                )
                if not step_text:
                    round_failed = True
                    self._log(
                        f"⚠️ Planned reasoning step {global_idx} could not be generated; "
                        "replanning residual steps from accepted prefix",
                        "warning",
                    )
                    break

                steps.append(step_text)
                step_summaries.append(self._compact_step_summary(step_text))
                new_norms = self._extract_cited_articles(step_text)
                for norm in new_norms:
                    if norm not in used_norms:
                        used_norms.append(norm)

                prec_mentions = [
                    p for p in allowed_precedents if p.lower() in step_text.lower()
                ]
                prec_info = (
                    f" | prec: {', '.join(prec_mentions)}" if prec_mentions else ""
                )
                self._log(
                    f"✅ Step {global_idx}: {step_text[:80]}... "
                    f"| norms: {', '.join(new_norms) if new_norms else 'none'}{prec_info}"
                )

                if len(steps) >= max_steps:
                    break

            if len(steps) == steps_before_round:
                stalled_rounds += 1
            else:
                stalled_rounds = 0

            if stalled_rounds >= 2 and len(steps) < min_steps:
                break

            if not round_failed and len(steps) >= min_steps:
                break

        if len(steps) < min_steps:
            raise RuntimeError(
                "Planner/executor produced fewer steps than chain_min_steps "
                "after residual replanning"
            )

        self._log(
            f"📊 Planned chain complete: {len(steps)} steps, "
            f"{len(set(used_norms))} unique norms"
        )
        return self._assemble_raw_response(claim, steps), steps

    def _generate_reasoning_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        anchor_text: str,
        principle_text: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        min_steps: int,
        max_steps: int,
        planner_mode: str = "FULL",
        resume_from_step: int = 1,
        existing_summaries: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Generate and validate an execution plan for support reasoning."""
        existing_steps_text = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(existing_summaries or [], start=1)
            )
            if existing_summaries
            else "- none"
        )
        prompt = render_prompt(
            "reasoner.generate_plan",
            claim=claim,
            routing_domain=routing_decision.domain,
            anchor_text=anchor_text,
            principle_text=principle_text,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            knowledge_base=knowledge_base,
            min_steps=min_steps,
            max_steps=max_steps,
            planner_mode=planner_mode,
            resume_from_step=resume_from_step,
            existing_steps=existing_steps_text,
        )
        last_error = "planner failed"
        for attempt in range(1, self._max_plan_retries + 1):
            try:
                resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
                raw = (resp.content or "").strip()
                plan = self._parse_reasoning_plan(
                    raw=raw,
                    min_steps=min_steps,
                    max_steps=max_steps,
                )
                if plan:
                    return plan
                last_error = "parsed empty plan"
            except Exception as e:
                last_error = str(e)
            self._log(
                f"⚠️ Planner attempt {attempt}/{self._max_plan_retries} failed: {last_error}",
                "warning",
            )
        raise RuntimeError(f"Support planner failed: {last_error}")

    def _parse_reasoning_plan(
        self,
        raw: str,
        min_steps: int,
        max_steps: int,
    ) -> list[dict[str, str]]:
        """Parse planner JSON and enforce plan quality constraints."""
        payload_text = raw.strip()
        if not payload_text.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", payload_text)
            if match:
                payload_text = match.group(0)
        data = json.loads(payload_text)
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list):
            raise ValueError("planner output missing 'steps' array")

        cleaned: list[dict[str, str]] = []
        for idx, item in enumerate(steps_raw, start=1):
            if not isinstance(item, dict):
                continue
            goal = str(item.get("goal", "")).strip()
            focus = str(item.get("focus", "")).strip()
            expected_norm = str(item.get("expected_norm", "")).strip() or "N/A"
            step_type = self._normalize_reasoner_step_type(item.get("step_type"))
            novelty_key = (
                re.sub(
                    r"[^a-z0-9_]+",
                    "_",
                    str(item.get("novelty_key", "")).strip().lower(),
                ).strip("_")
                or re.sub(r"[^a-z0-9_]+", "_", focus.lower()).strip("_")[:48]
                or f"step_{idx}"
            )
            citation_requirement = self._normalize_plan_citation_requirement(
                expected_norm=expected_norm,
                raw_value=item.get("citation_requirement"),
            )
            if not goal or not focus:
                continue
            cleaned.append(
                {
                    "id": str(item.get("id", f"P{idx}")).strip() or f"P{idx}",
                    "goal": goal,
                    "focus": focus,
                    "expected_norm": expected_norm,
                    "citation_requirement": citation_requirement,
                    "step_type": step_type,
                    "novelty_key": novelty_key[:64],
                }
            )

        if len(cleaned) < min_steps or len(cleaned) > max_steps:
            raise ValueError(
                f"invalid plan length {len(cleaned)} (expected {min_steps}-{max_steps})"
            )
        novelty_keys = [step.get("novelty_key", "") for step in cleaned]
        if len(set(novelty_keys)) != len(novelty_keys):
            raise ValueError("planner produced duplicate novelty_key values")
        if not self._has_min_plan_type_coverage(cleaned):
            raise ValueError("planner produced poor step-type coverage")
        if self._has_overlapping_plan_steps(cleaned):
            raise ValueError("planner produced overlapping/repetitive steps")
        return cleaned

    @staticmethod
    def _normalize_reasoner_step_type(raw_value: object) -> str:
        """Normalize planner step_type into a constrained enum-like value."""
        value = str(raw_value or "").strip().upper()
        allowed = {
            "FACTS",
            "QUALIFICATION",
            "CAUSAL_LINK",
            "ELEMENTS",
            "BALANCING",
            "CONSEQUENCE",
            "SYNTHESIS",
            "OTHER",
        }
        if value in allowed:
            return value
        aliases = {
            "FACTUAL": "FACTS",
            "FATTI": "FACTS",
            "CLASSIFICATION": "QUALIFICATION",
            "LEGAL_QUALIFICATION": "QUALIFICATION",
            "NEXUS": "CAUSAL_LINK",
            "CAUSALITY": "CAUSAL_LINK",
            "SUBJECTIVE_ELEMENT": "ELEMENTS",
            "PENAL_ELEMENT": "ELEMENTS",
            "OUTCOME": "CONSEQUENCE",
            "RESULT": "CONSEQUENCE",
            "CONCLUSION": "SYNTHESIS",
        }
        return aliases.get(value, "OTHER")

    @staticmethod
    def _has_min_plan_type_coverage(plan_steps: list[dict[str, str]]) -> bool:
        """Require at least minimal diversity of step types for non-trivial plans."""
        if len(plan_steps) <= 2:
            return True
        concrete = [
            str(step.get("step_type", "")).strip().upper()
            for step in plan_steps
            if str(step.get("step_type", "")).strip().upper() not in {"", "OTHER"}
        ]
        if len(concrete) < 2:
            # Backward compatibility when planner does not provide step_type.
            return True
        min_required = 2 if len(plan_steps) <= 4 else 3
        return len(set(concrete)) >= min_required

    @staticmethod
    def _normalize_plan_citation_requirement(
        *,
        expected_norm: str,
        raw_value: object,
    ) -> str:
        """
        Normalize planner citation policy for a step.

        Backward-compatible behavior:
        - if planner does not provide the field, require citations only when
          ``expected_norm`` is a real article (not N/A).
        """
        value = str(raw_value or "").strip().lower()
        aliases = {
            "required": "required",
            "must": "required",
            "mandatory": "required",
            "optional": "optional",
            "if_possible": "optional",
            "when_possible": "optional",
            "none": "none",
            "no": "none",
        }
        normalized = aliases.get(value)
        if normalized:
            return normalized

        expected = str(expected_norm or "").strip().upper()
        if expected and expected not in {"N/A", "NA", "NONE", "-"}:
            return "required"
        return "optional"

    @staticmethod
    def _step_requires_citation(
        *, expected_norm: str, citation_requirement: str
    ) -> bool:
        """Decide citation requirement using planner metadata only (no keyword heuristics)."""
        requirement = str(citation_requirement or "").strip().lower()
        if requirement == "required":
            return True
        if requirement in {"optional", "none"}:
            return False
        expected = str(expected_norm or "").strip().upper()
        return bool(expected and expected not in {"N/A", "NA", "NONE", "-"})

    @staticmethod
    def _extract_allowed_article_ids(allowed_statutes: list[str]) -> set[str]:
        """
        Parse normalized article ids from allowed statute labels.
        """
        ids: set[str] = set()
        for label in allowed_statutes or []:
            for match in re.findall(
                r"art\.?\s*([0-9]+(?:-[a-z]+)?)", label, re.IGNORECASE
            ):
                normalized = normalize_article_id(match)
                if normalized:
                    ids.add(normalized)
        return ids

    def _coerce_plan_to_allowed_norms(
        self,
        *,
        plan: list[dict[str, str]],
        allowed_statutes: list[str],
    ) -> list[dict[str, str]]:
        """
        Downgrade impossible citation requirements when planner expects norms
        not present in the allowed statute set.
        """
        allowed_ids = self._extract_allowed_article_ids(allowed_statutes)
        if not allowed_ids:
            return plan

        adjusted = 0
        for step in plan:
            expected_norm = str(step.get("expected_norm", "")).strip()
            if not expected_norm or expected_norm.upper() in {"N/A", "NA", "NONE", "-"}:
                continue
            mentions = extract_article_mentions(expected_norm, require_code=False)
            expected_ids = {
                normalize_article_id(m.article_id)
                for m in mentions
                if getattr(m, "article_id", None)
            }
            expected_ids = {eid for eid in expected_ids if eid}
            if not expected_ids:
                for match in re.findall(
                    r"art\.?\s*([0-9]+(?:-[a-z]+)?)", expected_norm, re.IGNORECASE
                ):
                    normalized = normalize_article_id(match)
                    if normalized:
                        expected_ids.add(normalized)
            if not expected_ids:
                continue
            if expected_ids & allowed_ids:
                continue
            step["expected_norm"] = "N/A"
            if str(step.get("citation_requirement", "")).strip().lower() == "required":
                step["citation_requirement"] = "optional"
            adjusted += 1

        if adjusted:
            self._log(
                f"⚠️ Planner normalization: downgraded {adjusted} step(s) with unavailable expected_norm",
                "warning",
            )
        return plan

    def _has_overlapping_plan_steps(self, plan_steps: list[dict[str, str]]) -> bool:
        """Detect overlap across planned goals/focuses (lexical + semantic)."""
        normalized = []
        texts: list[str] = []
        for step in plan_steps:
            text = f"{step.get('goal', '')} {step.get('focus', '')}".lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            words = {w for w in text.split() if len(w) > 3}
            normalized.append(words)
            texts.append(
                re.sub(
                    r"\s+",
                    " ",
                    f"{step.get('goal', '')}. {step.get('focus', '')}",
                ).strip()
            )

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                a = normalized[i]
                b = normalized[j]
                if not a or not b:
                    continue
                overlap = len(a & b) / len(a | b)
                if overlap >= 0.65:
                    return True
                if overlap >= 0.40:
                    rel_ab = self._nli_relation(
                        target_text=texts[i],
                        attacker_text=texts[j],
                        actor_label="ReasonerPlanner",
                    )
                    rel_ba = self._nli_relation(
                        target_text=texts[j],
                        attacker_text=texts[i],
                        actor_label="ReasonerPlanner",
                    )
                    if rel_ab == "entailment" and rel_ba == "entailment":
                        return True
        return False

    def _prune_plan_against_existing_history(
        self,
        *,
        plan: list[dict[str, str]],
        previous_summaries: list[str],
    ) -> list[dict[str, str]]:
        """Remove residual plan steps already covered by the accepted prefix."""
        if not plan or not previous_summaries:
            return plan
        pruned: list[dict[str, str]] = []
        removed = 0
        for step in plan:
            candidate = f"{step.get('goal', '')}. {step.get('focus', '')}".strip()
            if not candidate:
                removed += 1
                continue
            if self._is_repetitive_step(candidate, previous_summaries, threshold=0.45):
                removed += 1
                continue
            redundant = False
            for summary in previous_summaries[-4:]:
                rel = self._nli_relation(
                    target_text=summary,
                    attacker_text=candidate,
                    actor_label="ReasonerPlanner",
                )
                if rel == "entailment":
                    rel_sym = self._nli_relation(
                        target_text=candidate,
                        attacker_text=summary,
                        actor_label="ReasonerPlanner",
                    )
                    if rel_sym in {"entailment", "neutral"}:
                        redundant = True
                        break
            if redundant:
                removed += 1
                continue
            pruned.append(step)
        if removed:
            self._log(
                f"⚠️ Planner normalization: pruned {removed} residual step(s) already covered by accepted prefix",
                "warning",
            )
        return pruned

    def _generate_support_step_from_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        anchor_text: str,
        principle_text: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        plan: list[dict[str, str]],
        plan_index: int,
        plan_step: dict[str, str],
        previous_steps: list[str],
        previous_summaries: list[str],
        used_norms: list[str],
        stream_callback: Optional[Callable[[dict], None]],
    ) -> str:
        """Execute one planned support step with validation + retries."""
        last_candidate = ""
        last_reason = "invalid output"
        for attempt in range(1, self._max_step_rewrites + 2):
            prompt = (
                self._build_support_step_prompt_from_plan(
                    claim=claim,
                    routing_decision=routing_decision,
                    anchor_text=anchor_text,
                    principle_text=principle_text,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    precedents_list=precedents_list,
                    plan=plan,
                    plan_index=plan_index,
                    plan_step=plan_step,
                    previous_summaries=previous_summaries,
                    used_norms=used_norms,
                )
                if attempt == 1
                else self._build_support_plan_rewrite_prompt(
                    previous_prompt=self._build_support_step_prompt_from_plan(
                        claim=claim,
                        routing_decision=routing_decision,
                        anchor_text=anchor_text,
                        principle_text=principle_text,
                        knowledge_base=knowledge_base,
                        statutes_list=statutes_list,
                        precedents_list=precedents_list,
                        plan=plan,
                        plan_index=plan_index,
                        plan_step=plan_step,
                        previous_summaries=previous_summaries,
                        used_norms=used_norms,
                    ),
                    invalid_step=last_candidate,
                    invalid_reason=last_reason,
                )
            )
            try:
                resp = self._resilient_llm_invoke(
                    [HumanMessage(content=prompt)],
                    stream_callback=(
                        (
                            lambda token: self._emit_stream_token(
                                stream_callback,
                                phase="support",
                                token=token,
                                step=plan_index,
                            )
                        )
                        if stream_callback and attempt == 1
                        else None
                    ),
                )
                candidate = self._parse_step_text((resp.content or "").strip())
            except PipelineCancelled:
                raise
            except Exception as exc:
                last_reason = f"generation error: {exc}"
                self._log(
                    f"⚠️ Step {plan_index} generation failed (attempt {attempt}): {exc}",
                    "warning",
                )
                continue

            last_candidate = candidate
            ok, reason = self._validate_support_step_candidate(
                candidate_step=candidate,
                previous_steps=previous_steps,
                claim=claim,
                expected_norm=plan_step.get("expected_norm", "N/A"),
                citation_requirement=plan_step.get("citation_requirement", "optional"),
            )
            if ok:
                return candidate
            last_reason = reason
            if stream_callback:
                try:
                    stream_callback(
                        {
                            "phase": "support",
                            "action": "reset_step",
                            "step": plan_index,
                        }
                    )
                except PipelineCancelled:
                    raise
                except Exception:
                    pass
            self._log(
                f"⚠️ Step {plan_index} rejected ({reason}) "
                f"[attempt {attempt}/{self._max_step_rewrites + 1}]",
                "warning",
            )
        return ""

    def _build_support_step_prompt_from_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        anchor_text: str,
        principle_text: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        plan: list[dict[str, str]],
        plan_index: int,
        plan_step: dict[str, str],
        previous_summaries: list[str],
        used_norms: list[str],
    ) -> str:
        """Create prompt for one planned reasoning step."""
        plan_lines = "\n".join(
            f"{idx}. {step.get('goal', '')} | focus: {step.get('focus', '')} | "
            f"type: {step.get('step_type', 'OTHER')} | novelty: {step.get('novelty_key', '')}"
            for idx, step in enumerate(plan, start=1)
        )
        summary_lines = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(previous_summaries, start=1)
            )
            if previous_summaries
            else "- none"
        )
        used_norms_text = ", ".join(used_norms) if used_norms else "none"
        return render_prompt(
            "reasoner.support_step",
            claim=claim,
            routing_domain=routing_decision.domain,
            anchor_text=anchor_text,
            principle_text=principle_text,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            plan_lines=plan_lines,
            plan_index=plan_index,
            plan_goal=plan_step.get("goal", ""),
            plan_focus=plan_step.get("focus", ""),
            plan_expected_norm=plan_step.get("expected_norm", "N/A"),
            plan_citation_requirement=plan_step.get("citation_requirement", "optional"),
            plan_step_type=plan_step.get("step_type", "OTHER"),
            plan_novelty_key=plan_step.get("novelty_key", ""),
            summary_lines=summary_lines,
            used_norms_text=used_norms_text,
        )

    def _build_support_plan_rewrite_prompt(
        self, previous_prompt: str, invalid_step: str, invalid_reason: str
    ) -> str:
        """Prompt to rewrite a planned step that failed validation."""
        base_prompt = render_prompt(
            "reasoner.support_plan_rewrite",
            previous_prompt=previous_prompt,
            invalid_reason=invalid_reason,
            invalid_step=invalid_step,
        )
        if "missing statutory citation" in (invalid_reason or "").lower():
            return (
                f"{base_prompt}\n\n"
                "MANDATORY FIX:\n"
                "- Include at least one explicit statute citation from the ALLOWED STATUTES list already shown above.\n"
                "- Use the citation in canonical form (e.g., Art. 589-bis c.p.).\n"
            )
        return base_prompt

    def _validate_support_step_candidate(
        self,
        candidate_step: str,
        previous_steps: list[str],
        claim: str,
        expected_norm: str,
        citation_requirement: str,
    ) -> tuple[bool, str]:
        """Validation checks for one reasoning step candidate."""
        text = (candidate_step or "").strip()
        if not text or text.upper() == "DONE":
            return False, "empty step"
        if self._is_garbage_text(text):
            return False, "garbage output"
        if not self._is_support_step_consistent(text):
            return False, "reasoning inconsistency"
        facts_ok, facts_reason = self._is_step_fact_consistent_with_claim(
            claim=claim,
            candidate_step=text,
            actor_label="Reasoner",
        )
        if not facts_ok:
            return False, facts_reason
        compatible, compat_reason = self._is_support_step_compatible_with_history(
            candidate_step=text,
            previous_steps=previous_steps,
            claim=claim,
        )
        if not compatible:
            return False, compat_reason or "history incompatibility"
        citations = self._extract_cited_articles(text)
        if (
            self._step_requires_citation(
                expected_norm=expected_norm,
                citation_requirement=citation_requirement,
            )
            and not citations
        ):
            return False, "missing statutory citation"
        if previous_steps and self._is_repetitive_step(text, previous_steps):
            return False, "lexical repetition"
        if previous_steps and self._is_semantically_redundant_step(
            candidate_step=text,
            previous_steps=previous_steps,
            claim=claim,
            role="support",
        ):
            return False, "semantic repetition"
        return True, ""

    def _is_semantically_redundant_step(
        self,
        candidate_step: str,
        previous_steps: list[str],
        claim: str,
        role: str,
    ) -> bool:
        """LLM-based semantic redundancy check (NEW vs REPEAT)."""
        if not previous_steps:
            return False
        context_prev = "\n".join(
            f"{idx}. {step}" for idx, step in enumerate(previous_steps[-3:], start=1)
        )
        prompt = render_prompt(
            "reasoner.semantic_redundancy",
            claim=claim,
            role=role,
            context_prev=context_prev,
            candidate_step=candidate_step,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip().upper()
            return "REPEAT" in answer
        except Exception:
            return False

    @staticmethod
    def _compact_step_summary(step_text: str) -> str:
        """Compact summary used as execution memory for following steps."""
        first_sentence = re.split(r"(?<=[.!?])\s+", step_text.strip())[0]
        first_sentence = re.sub(r"\s+", " ", first_sentence).strip()
        return first_sentence[:220]

    @staticmethod
    def _emit_stream_token(
        stream_callback: Optional[Callable[[dict], None]],
        *,
        phase: str,
        token: str,
        step: Optional[int] = None,
    ) -> None:
        """Emit one token chunk to external streaming callback."""
        if not stream_callback or not token:
            return
        payload: dict[str, str | int] = {"phase": phase, "token": token}
        if step is not None:
            payload["step"] = step
        try:
            stream_callback(payload)
        except PipelineCancelled:
            raise
        except Exception:
            # Streaming callback errors must never break reasoning generation.
            pass

    def _parse_step_text(self, response: str) -> str:
        """Extract the step text from an LLM response.

        Looks for a ``STEP:`` / ``STEP N:`` (or ``PASSO:``) marker and
        returns everything after it.  Falls back to the full response
        if no marker is found.

        Also removes any leading numeric prefixes (e.g., "3 Per valutare...")
        that the LLM may have included.
        """
        lines = response.strip().split("\n")
        step_lines: list[str] = []
        found_step = False

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            # Match STEP:, STEP 1:, STEP 12:, PASSO:, PASSO 1: etc.
            if re.match(r"^(STEP|PASSO)\s*\d*\s*:", upper):
                content = re.sub(
                    r"^(?:STEP|PASSO)\s*\d*\s*:\s*",
                    "",
                    stripped,
                    flags=re.IGNORECASE,
                ).strip()
                if content:
                    step_lines.append(content)
                found_step = True
                continue

            if found_step and stripped:
                step_lines.append(stripped)

        step_text = " ".join(step_lines).strip()

        # Fallback: use entire response (skip DECISION / STEP N: prefixes)
        if not step_text:
            fallback_lines = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                su = s.upper()
                if su.startswith("DECISION:") or su.startswith("DECISIONE:"):
                    continue
                # Strip any leading STEP N: prefix
                s = re.sub(
                    r"^(?:STEP|PASSO)\s*\d*\s*:\s*",
                    "",
                    s,
                    flags=re.IGNORECASE,
                )
                if s:
                    fallback_lines.append(s)
            step_text = " ".join(fallback_lines).strip()

        # P6 FIX: Remove leading numeric prefixes like "3 Per valutare..."
        # This handles cases where LLM responds with just "3 Per valutare" without STEP:
        step_text = re.sub(r"^\d+\s+", "", step_text)

        return step_text

    def _build_support_conclusion_rewrite_prompt(
        self, claim: str, chain_text: str, norms_text: str, invalid_conclusion: str
    ) -> str:
        """Force a concise conclusion aligned with the reasoning chain."""
        return render_prompt(
            "reasoner.support_conclusion_rewrite",
            claim=claim,
            chain_text=chain_text,
            norms_text=norms_text,
            invalid_conclusion=invalid_conclusion,
        )

    def _is_support_step_consistent(self, step_text: str) -> bool:
        """
        Lightweight guardrail against self-contradictory legal reasoning text.

        This intentionally does NOT enforce a pro/anti orientation on the claim.
        It only rejects obvious internal contradictions inside a single step.
        """
        return self._is_step_self_consistent(step_text)

    def _is_support_step_compatible_with_history(
        self,
        *,
        candidate_step: str,
        previous_steps: list[str],
        claim: str,
    ) -> tuple[bool, str]:
        """
        Ensure candidate step is semantically compatible with accepted chain history.
        """
        if not previous_steps:
            return True, ""
        window = previous_steps[-3:]
        start_idx = len(previous_steps) - len(window) + 1
        for offset, prev_step in enumerate(window):
            relation = self._nli_relation(
                target_text=prev_step,
                attacker_text=candidate_step,
                actor_label="Reasoner",
            )
            if relation != "contradiction":
                continue
            # Reject only when contradiction is stable in both directions.
            relation_sym = self._nli_relation(
                target_text=candidate_step,
                attacker_text=prev_step,
                actor_label="Reasoner",
            )
            if relation_sym == "contradiction":
                step_no = start_idx + offset
                return False, f"candidate contradicts previous step {step_no}"
        return True, ""

    def _generate_conclusion(
        self,
        claim: str,
        steps: list[str],
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> str:
        """Generate a dynamic conclusion via LLM based on the reasoning chain.

        Returns a concise 2-4 sentence conclusion synthesizing the legal
        verdict from the chain steps.
        """
        if not steps:
            return ""

        chain_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        norms = self._extract_cited_articles(" ".join(steps))
        norms_text = ", ".join(norms) if norms else "le norme applicabili"

        prompt = render_prompt(
            "reasoner.generate_conclusion",
            claim=claim,
            chain_text=chain_text,
            norms_text=norms_text,
        )

        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                stream_callback=(
                    (
                        lambda token: self._emit_stream_token(
                            stream_callback,
                            phase="support_conclusion",
                            token=token,
                        )
                    )
                    if stream_callback
                    else None
                ),
            )
            conclusion = (resp.content or "").strip()
            # Clean up any echoed prefix
            conclusion = re.sub(
                r"^(?:CONCLUSIONE|CONCLUSION)\s*:\s*",
                "",
                conclusion,
                flags=re.IGNORECASE,
            ).strip()
            if conclusion:
                if not self._is_support_step_consistent(conclusion):
                    self._log(
                        "Warning: LLM conclusion appears internally inconsistent; trying rewrite (non-blocking)",
                        "warning",
                    )
                    original_conclusion = conclusion
                    rewrite_prompt = self._build_support_conclusion_rewrite_prompt(
                        claim=claim,
                        chain_text=chain_text,
                        norms_text=norms_text,
                        invalid_conclusion=conclusion,
                    )
                    try:
                        rewrite_resp = self._resilient_llm_invoke(
                            [HumanMessage(content=rewrite_prompt)]
                        )
                        rewritten = (rewrite_resp.content or "").strip()
                        rewritten = re.sub(
                            r"^(?:CONCLUSIONE|CONCLUSION)\s*:\s*",
                            "",
                            rewritten,
                            flags=re.IGNORECASE,
                        ).strip()
                        if rewritten:
                            conclusion = rewritten
                            if not self._is_support_step_consistent(conclusion):
                                self._log(
                                    "Warning: rewritten conclusion is still internally inconsistent; keeping rewritten anyway (fact-lock enforced)",
                                    "warning",
                                )
                        else:
                            self._log(
                                "Warning: rewritten conclusion is empty; keeping original conclusion",
                                "warning",
                            )
                            conclusion = original_conclusion
                    except Exception as rewrite_exc:
                        self._log(
                            f"Warning: conclusion rewrite failed: {rewrite_exc}; keeping original conclusion",
                            "warning",
                        )
                        conclusion = original_conclusion
                if conclusion:
                    facts_ok, _ = self._is_step_fact_consistent_with_claim(
                        claim=claim,
                        candidate_step=conclusion,
                        actor_label="Reasoner",
                    )
                    if not facts_ok:
                        self._log(
                            "Warning: LLM conclusion contradicts explicit claim facts; using fallback",
                            "warning",
                        )
                        conclusion = ""
                if conclusion:
                    self._log(f"LLM-generated conclusion: {conclusion[:120]}...")
                    return conclusion
        except PipelineCancelled:
            raise
        except Exception as e:
            self._log(f"⚠️ Conclusion generation failed: {e}", "warning")

        # Static fallback (neutral with respect to claim orientation)
        return (
            f"Sulla base dell'analisi giuridica svolta, la valutazione del claim "
            f"deve essere fondata sulle norme richiamate ({norms_text}) "
            f"e sulla loro applicazione ai fatti esposti."
        )

    def _assemble_raw_response(
        self,
        claim: str,
        steps: list[str],
        conclusion_text: str = "",
    ) -> str:
        """Assemble a complete raw response from iterative steps.

        Produces the same format expected by ``_extract_arguments`` and
        ``_extract_reasoning_chain``, so ``AspicFormatter`` works
        without changes.

        If *conclusion_text* is provided (LLM-generated), it is used
        directly; otherwise a static fallback is emitted.
        """
        chain_section = "**Catena di ragionamento**:\n"
        for i, step in enumerate(steps, 1):
            chain_section += f"{i}. {step}\n"

        premise_text = " ".join(steps)
        norms = self._extract_cited_articles(" ".join(steps))
        norms_text = "\n".join(f"- {n}" for n in norms) if norms else "N/D"

        if conclusion_text:
            if not self._is_support_step_consistent(conclusion_text):
                self._log(
                    "Warning: provided conclusion appears internally inconsistent (non-blocking); enforcing only fact-lock",
                    "warning",
                )
            facts_ok, _ = self._is_step_fact_consistent_with_claim(
                claim=claim,
                candidate_step=conclusion_text,
                actor_label="Reasoner",
            )
            if not facts_ok:
                self._log(
                    "Warning: provided conclusion contradicts explicit claim facts; using fallback",
                    "warning",
                )
                conclusion_text = ""

        if not conclusion_text:
            norms_list = ", ".join(norms) if norms else "le norme applicabili"
            conclusion_text = (
                f"Sulla base dell'analisi giuridica svolta, la valutazione del claim "
                f"dipende dall'applicazione delle norme richiamate ({norms_list}) "
                f"ai fatti esposti nella catena di ragionamento."
            )

        raw = (
            f"**Premessa**: {premise_text}\n\n"
            f"**Norma**:\n{norms_text}\n\n"
            f"**Nesso Causale**: La connessione tra le norme citate e la questione giuridica posta dal claim "
            f"emerge dalla catena di ragionamento sottostante, dove ciascun passo "
            f"costruisce logicamente sul precedente per motivare la valutazione "
            f"giuridica finale.\n\n"
            f"**Conclusione**: {conclusion_text}\n\n"
            f"{chain_section}"
        )
        return raw

    def _build_reasoning_prompt_with_context(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        anchor_text: str,
        principle_text: str,
        knowledge_base: str,
        allowed_statutes: list[str],
        allowed_precedents: list[str],
    ) -> str:
        """Build the prompt for the reasoning task with pre-retrieved context."""
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )
        return render_prompt(
            "reasoner.reasoning_with_context",
            claim=claim,
            routing_domain=routing_decision.domain,
            anchor_text=anchor_text,
            principle_text=principle_text,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            allowed_statutes_count=len(allowed_statutes),
        )

    def _format_anchor_norms(self, anchor_norms: dict) -> str:
        """Format anchor norms for prompt readability."""
        core = anchor_norms.get("core_norms", []) if anchor_norms else []
        accessory = anchor_norms.get("accessory_norms", []) if anchor_norms else []
        lines = []
        for n in core:
            # Support both old keys (riferimento/nota) and new keys (ref/role)
            ref = n.get("ref") or n.get("riferimento", "N/D")
            role = n.get("role") or n.get("nota", "")
            lines.append(f"- [core] {ref}: {role}")
        for n in accessory:
            ref = n.get("ref") or n.get("riferimento", "N/D")
            role = n.get("role") or n.get("nota", "")
            lines.append(f"- [accessory] {ref}: {role}")
        return "\n".join(lines) or "- No anchor norms defined"

    def _format_principle_tests(self, principle_tests: list[dict]) -> str:
        """Format principle tests list."""
        if not principle_tests:
            return "- No principle tests defined"
        lines = []
        for t in principle_tests:
            lines.append(
                f"- {t.get('id', 'TEST')} | {t.get('name', '')}: {t.get('description', '')}"
            )
        return "\n".join(lines)

    def _expand_with_cross_references(self, statutes: list[dict]) -> list[dict]:
        """
        Add statutes explicitly referenced inside the text of already-retrieved articles.

        Forward-only: parses each article's text for patterns like
        "art. 624", "artt. 1223" and fetches the cited article by exact
        number from the KB via get_statute_by_article_tool.
        """
        try:
            import re
        except Exception:
            return statutes

        seen = {(s.get("articolo"), s.get("source")) for s in statutes}
        extra: list[dict] = []

        pattern = re.compile(r"art(?:t)?\.?\s*(\d{2,4})", re.IGNORECASE)

        for s in statutes:
            text = s.get("testo") or ""
            source = s.get("source", "codice_civile")

            refs = set(pattern.findall(text))

            added_refs: list[str] = []
            for ref in refs:
                key = (ref, source)
                if key in seen:
                    continue
                seen.add(key)
                result = get_statute_by_article_tool.invoke(
                    {"articolo": ref, "codice": source}
                )
                if result and not result.get("error") and result.get("articolo"):
                    extra.append(result)
                    added_refs.append(ref)

            if added_refs:
                self._log(
                    f"🔗 Cross-ref from Art. {s.get('articolo')} -> {', '.join(sorted(set(added_refs)))}"
                )

        # Deduplicate and merge
        merged = statutes + extra
        deduped: list[dict] = []
        seen_final = set()
        for st in merged:
            k = (st.get("articolo"), st.get("source"))
            if k in seen_final:
                continue
            seen_final.add(k)
            deduped.append(st)
        return deduped

    def _extract_arguments(self, response: str) -> list[dict]:
        """Extract structured arguments from response."""
        arguments = []

        # Simple extraction based on keywords
        sections = response.split("**")
        current_arg: dict[str, str] = {}

        for i, section in enumerate(sections):
            section = section.strip()
            lower_section = section.lower()

            if "premessa" in lower_section:
                if current_arg:
                    arguments.append(current_arg)
                current_arg = {"type": "premise"}
            elif "norma" in lower_section:
                current_arg["type"] = "norm"
            elif "nesso" in lower_section:
                current_arg["type"] = "link"
            elif "conclusione" in lower_section:
                current_arg["type"] = "conclusion"
            elif current_arg and i > 0:
                key = current_arg.get("type", "content")
                current_arg[key] = section

        if current_arg:
            arguments.append(current_arg)

        return arguments
