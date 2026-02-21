"""
LexCausa Reasoner Agent.

The Reasoner is responsible for:
1. Receiving a legal claim with pre-retrieved statutes and precedents
2. Generating supporting arguments based on the provided knowledge base
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
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_statute_by_article_tool
from .tools.taxonomy_tools import get_causality_theory_tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_react_invoke  # noqa: E402

# System prompt for the Reasoner (with pre-retrieved context)
REASONER_SYSTEM_PROMPT = """IMPORTANT: You MUST respond ENTIRELY in Italian. Every word of your response must be in Italian.

You are the Reasoner. The router already set causal_type_id and theory_id.
Do NOT re-classify. Use these as structural constraints:
- anchor_norms (core + accessory) from config
- principle_tests for the causal type

You receive a pre-retrieved KNOWLEDGE BASE (statutes/precedents) filtered as supportive/neutral.
Build ONLY supporting arguments for the claim using the provided sources.

Critical rules:
- Cite ONLY statutes and precedents present in the KNOWLEDGE BASE.
- If a needed statute is missing, state "articolo non disponibile nella knowledge base".
- Keep reasoning independent: do not reference the Counter-Reasoner.
- Use ONLY facts explicitly stated in the claim; do NOT invent, assume, or complete missing facts.
- Your response MUST end with a **Catena di ragionamento**: section containing a numbered list.
- Numbered lists (1. 2. 3. ...) are ONLY allowed inside **Catena di ragionamento**. Use prose or bullet points ("-") everywhere else.
- MANDATORY: Your ENTIRE response must be written in Italian. Do NOT write in English."""


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
    classifies causality type, and generates supporting arguments.

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
        self._max_support_stance_rewrites = 1
        self._max_plan_retries = 3
        self._max_step_rewrites = 2

    def _resilient_model_order(self) -> list[str] | None:
        """Reasoner fallback chain: selected model, then OSS -> Maverick -> Scout."""
        preferred_chain = [
            settings.resolve_model_name("gpt_oss_120b"),
            settings.resolve_model_name("groq_llama_maverick_17b"),
            settings.resolve_model_name("groq_llama_scout_17b"),
        ]
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
        1) Generate initial reasoning from claim + supportive/neutral sources.
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

        if enable_causality:
            # Phase 1: initial reasoning (no anchor injection)
            base_statutes = self._expand_with_cross_references(pre_retrieved_statutes)
            kb1 = self._format_context_for_prompt(
                base_statutes, pre_retrieved_precedents
            )
            allowed_statutes1 = [
                f"Art. {s.get('articolo')} ({self._source_short_label(s.get('source', ''))})"
                for s in base_statutes
            ]
            allowed_precedents1 = [
                p.get("title", "Untitled") for p in pre_retrieved_precedents
            ]

            input_prompt1 = self._build_reasoning_prompt_with_context(
                claim,
                routing_decision,
                anchor_text="-",
                principle_text="-",
                knowledge_base=kb1,
                allowed_statutes=allowed_statutes1,
                allowed_precedents=allowed_precedents1,
            )

            raw_output1, _ = self._invoke_reasoner(input_prompt1)
            reasoning_chain1 = self._extract_reasoning_chain(raw_output1)

            # Phase 2: classify causality on reasoning chain filtered by domain
            chain_class = self._classify_causality_from_reasoning(
                claim, reasoning_chain1, raw_output1, domain
            )

            # DEBUG: log chain classification results
            self._log(
                f"🔬 Chain classification: causal_type_id={chain_class.get('causal_type_id')}, theory_id={chain_class.get('theory_id')}"
            )

            final_causal_id = chain_class.get("causal_type_id") or ""
            final_theory_id = chain_class.get("theory_id") or ""

            # Validate and get anchor norms for the classified causal type
            if final_causal_id:
                final_causal_id, final_theory_id = config_loader.validate_ids(
                    final_causal_id, final_theory_id
                )

            causal_types_for_counter: list[str] = (
                [final_causal_id] if final_causal_id else []
            )
            anchor_norms: dict = {}
            anchor_statutes: list[dict] = []
            principle_tests: list[dict] = []

            if final_causal_id:
                anchor_norms, anchor_statutes, principle_tests = (
                    self._filtered_anchor_norms_for_types([final_causal_id], claim)
                )
                self._log(
                    f"📋 Anchor norms retrieved: core={len(anchor_norms.get('core_norms', []))}, accessory={len(anchor_norms.get('accessory_norms', []))}"
                )
                self._log(f"📋 Anchor statutes to inject: {len(anchor_statutes)}")
        else:
            self._log(
                "🔬 Causality DISABLED — skipping classification and anchor norms"
            )
            chain_class = {}
            final_causal_id = ""
            final_theory_id = ""
            causal_types_for_counter = []
            anchor_norms = {}
            anchor_statutes = []
            principle_tests = []

        # Phase 3: refine reasoning with anchor norms + cross-ref expansion
        self._log(
            f"📌 Phase 3: merging pre_retrieved ({len(pre_retrieved_statutes)}) + anchor ({len(anchor_statutes)}) statutes"
        )
        all_statutes = pre_retrieved_statutes + anchor_statutes
        seen_keys = set()
        deduped_statutes = []
        for s in all_statutes:
            key = (s.get("articolo"), s.get("source"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_statutes.append(s)
        self._log(f"📌 After dedup: {len(deduped_statutes)} statutes")
        before_expand = len(deduped_statutes)
        deduped_statutes = self._expand_with_cross_references(deduped_statutes)
        if len(deduped_statutes) > before_expand:
            self._log(
                f"➕ Added {len(deduped_statutes) - before_expand} statutes via cross-ref",
                "info",
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

            output = ReasonerOutput(
                claim=claim,
                causality_classification={
                    **chain_class,
                    "domain": domain,
                    "source": "reasoning_chain",
                },
                causal_type_id=final_causal_id,
                theory_id=final_theory_id or "",
                causal_type_ids_for_counter=causal_types_for_counter,
                mismatch_status="",
                anchor_norms=anchor_norms,
                principle_tests=principle_tests,
                relevant_statutes=deduped_statutes,
                relevant_precedents=pre_retrieved_precedents,
                raw_response=raw_output,
                conclusion=conclusion,
            )

            # Use iterative chain directly; fall back to extraction if empty
            output.reasoning_chain = (
                iterative_chain
                if iterative_chain
                else self._extract_reasoning_chain(raw_output)
            )
            output.arguments = self._extract_arguments(raw_output)
            output.reasoning_chain = self._sanitize_reasoning_chain(
                output.reasoning_chain, pre_retrieved_precedents
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
        # Pattern per articoli: "Art. 2043 c.c.", "art. 1223 c.p.", "art. 10 L. 241/1990", etc.
        pattern = re.compile(
            r"(?:art(?:icolo)?\.?\s*)(\d{1,4}(?:[-\s]?[a-z]+)?)\s*"
            r"(c\.?[cp]\.?|"
            r"cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?|amm(?:inistrativ[oa])?)|"
            r"l(?:egge)?\.?\s*241\s*/?\s*1990|"
            r"241\s*/?\s*1990)",
            re.IGNORECASE,
        )
        matches = pattern.findall(text)
        articles = []
        for num, code in matches:
            code_lower = code.lower()
            if "241" in code_lower or "amm" in code_lower:
                code_norm = "L. 241/1990"
            elif "civ" in code_lower or "c.c" in code_lower:
                code_norm = "c.c."
            else:
                code_norm = "c.p."
            articles.append(f"Art. {num} {code_norm}")
        # Deduplica mantenendo ordine
        seen = set()
        unique = []
        for a in articles:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

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

        # Estrai gli articoli citati dalla catena di ragionamento
        cited_articles = self._extract_cited_articles(chain_text)
        articles_text = (
            ", ".join(cited_articles) if cited_articles else "Nessun articolo citato"
        )

        self._log(f"🔬 Articoli citati nella catena di ragionamento: {articles_text}")

        # Build causal type descriptions for prompt
        type_descriptions = []
        for ct_id in allowed_ids:
            ct = ct_index.get(ct_id, {})
            type_descriptions.append(
                f"- {ct_id}: {ct.get('name', '')} [{ct.get('domain', '')}]"
            )

        prompt = f"""You are a classifier. Based PRIMARILY on the CITED ARTICLES from the reasoning chain, choose the most appropriate causal_type_id.

Allowed causal_type_id values (domain={domain}):
{chr(10).join(type_descriptions)}

Classification criteria (based on cited articles):
- If articles are from codice civile (c.c.) like Art. 2043, 2056, 1223, 1226, 1227 → civil causality types
- If articles are from codice penale (c.p.) like Art. 40, 41 → criminal causality types
- If articles are from L. 241/1990 / procedimento amministrativo (e.g. Art. 1, 2, 3, 10-bis, 21-octies) → administrative causality/procedural types
- Consider the combination of articles to determine the most specific causal type

If uncertain, choose the closest from the allowed list.
Respond with ONLY the causal_type_id (no JSON, no explanation, just the id).

ORIGINAL CLAIM (for context only):
{claim}

CITED ARTICLES FROM REASONING (primary classification basis):
{articles_text}

REASONING CHAIN (for context):
{chain_text}
"""
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
        """Generate the reasoning chain with plan -> execute workflow.

        Flow:
        1. Build a reasoning plan (distinct steps) with one LLM call.
        2. Execute one LLM call per planned step.
        3. Validate each produced step (stance, repetition, semantic novelty).

        No fallback to the previous auto-stop strategy is used.
        """
        max_steps = settings.chain_max_steps
        min_steps = settings.chain_min_steps
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )

        plan = self._generate_reasoning_plan(
            claim=claim,
            routing_decision=routing_decision,
            anchor_text=anchor_text,
            principle_text=principle_text,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        self._log(f"🧭 Reasoning plan generated: {len(plan)} step(s)")

        steps: list[str] = []
        step_summaries: list[str] = []
        used_norms: list[str] = []

        for idx, plan_step in enumerate(plan, start=1):
            self._log(
                f"🔗 Generating planned step {idx}/{len(plan)}: "
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
                plan_index=idx,
                plan_step=plan_step,
                previous_steps=steps,
                previous_summaries=step_summaries,
                used_norms=used_norms,
                stream_callback=stream_callback,
            )
            if not step_text:
                raise RuntimeError(
                    f"Planned support step {idx} could not be generated with valid content"
                )

            steps.append(step_text)
            step_summaries.append(self._compact_step_summary(step_text))
            new_norms = self._extract_cited_articles(step_text)
            for norm in new_norms:
                if norm not in used_norms:
                    used_norms.append(norm)

            prec_mentions = [
                p for p in allowed_precedents if p.lower() in step_text.lower()
            ]
            prec_info = f" | prec: {', '.join(prec_mentions)}" if prec_mentions else ""
            self._log(
                f"✅ Step {idx}: {step_text[:80]}... "
                f"| norms: {', '.join(new_norms) if new_norms else 'none'}{prec_info}"
            )

        if len(steps) < min_steps:
            raise RuntimeError(
                "Planner/executor produced fewer steps than chain_min_steps"
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
    ) -> list[dict[str, str]]:
        """Generate and validate an execution plan for support reasoning."""
        prompt = f"""You are a legal planning engine for Italian law.

Create a step-by-step plan to SUPPORT the claim.
The plan must be executable in sequence and each step must be materially different.
Return ONLY valid JSON (no markdown, no prose) with this schema:
{{
  "steps": [
    {{
      "id": "P1",
      "goal": "specific legal objective for this step",
      "focus": "single legal/factual focus",
      "expected_norm": "article expected to be cited or 'N/A'"
    }}
  ]
}}

CLAIM:
"{claim}"

DOMAIN: {routing_decision.domain}
ANCHOR NORMS:
{anchor_text}

PRINCIPLE TESTS:
{principle_text}

ALLOWED STATUTES:
{statutes_list}

ALLOWED PRECEDENTS:
{precedents_list}

KNOWLEDGE BASE:
{knowledge_base}

RULES:
- Number of steps must be between {min_steps} and {max_steps}.
- Each step must address a DIFFERENT objective (no overlap/rephrasing).
- Steps must be ordered logically (premise -> legal qualification -> applicability -> consequence -> final support).
- Every step must be pro-claim.
- Use only facts explicitly present in claim (no assumptions, no hypothetical factual completions).
- Prefer using different statutes across steps when possible.
- Keep each 'goal' and 'focus' concise (max 25 words each).
"""
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
            if not goal or not focus:
                continue
            cleaned.append(
                {
                    "id": str(item.get("id", f"P{idx}")).strip() or f"P{idx}",
                    "goal": goal,
                    "focus": focus,
                    "expected_norm": expected_norm,
                }
            )

        if len(cleaned) < min_steps or len(cleaned) > max_steps:
            raise ValueError(
                f"invalid plan length {len(cleaned)} (expected {min_steps}-{max_steps})"
            )
        if self._has_overlapping_plan_steps(cleaned):
            raise ValueError("planner produced overlapping/repetitive steps")
        return cleaned

    def _has_overlapping_plan_steps(self, plan_steps: list[dict[str, str]]) -> bool:
        """Detect obvious overlap across planned goals/focuses."""
        normalized = []
        for step in plan_steps:
            text = f"{step.get('goal', '')} {step.get('focus', '')}".lower()
            text = re.sub(r"[^a-z0-9àèéìòù\s]", " ", text)
            words = {w for w in text.split() if len(w) > 3}
            normalized.append(words)

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                a = normalized[i]
                b = normalized[j]
                if not a or not b:
                    continue
                overlap = len(a & b) / len(a | b)
                if overlap >= 0.65:
                    return True
        return False

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
            )
            if ok:
                return candidate
            if (
                reason == "semantic repetition"
                and attempt == self._max_step_rewrites + 1
                and candidate
                and self._is_support_step_consistent(candidate)
                and not self._is_garbage_text(candidate)
                and (
                    not previous_steps
                    or not self._is_repetitive_step(candidate, previous_steps)
                )
            ):
                self._log(
                    f"⚠️ Step {plan_index}: accepting semantically-close step after retries",
                    "warning",
                )
                return candidate
            last_reason = reason
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
        """Create prompt for one planned support step."""
        plan_lines = "\n".join(
            f"{idx}. {step.get('goal', '')} | focus: {step.get('focus', '')}"
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
        return f"""You are an expert Italian jurist.
You must execute ONLY one planned SUPPORT step, keeping strict pro-claim stance.

CLAIM:
"{claim}"

DOMAIN: {routing_decision.domain}
ANCHOR NORMS:
{anchor_text}
PRINCIPLE TESTS:
{principle_text}

KNOWLEDGE BASE (use only these sources):
{knowledge_base}

ALLOWED STATUTES:
{statutes_list}
ALLOWED PRECEDENTS:
{precedents_list}

GLOBAL PLAN:
{plan_lines}

CURRENT STEP TO EXECUTE: {plan_index}
- Goal: {plan_step.get("goal", "")}
- Focus: {plan_step.get("focus", "")}
- Expected norm: {plan_step.get("expected_norm", "N/A")}

ALREADY GENERATED STEP SUMMARIES:
{summary_lines}

NORMS ALREADY USED: {used_norms_text}

HARD RULES:
- Generate EXACTLY ONE atomic step in Italian (2-4 sentences).
- It must advance the plan and add NEW information, not paraphrase prior steps.
- It must support the claim only (no doubts, no balancing, no anti-claim hints).
- Use only facts explicitly in claim.
- Do not infer unprovided facts (no assumptions, no hypothetical completions of the factual scenario).
- Cite at least one statute when legally possible.
- If citing a precedent, include its full exact title from allowed list.

RESPONSE FORMAT:
STEP: [italian atomic step]
"""

    def _build_support_plan_rewrite_prompt(
        self, previous_prompt: str, invalid_step: str, invalid_reason: str
    ) -> str:
        """Prompt to rewrite a planned step that failed validation."""
        return (
            f"{previous_prompt}\n\n"
            "YOUR PREVIOUS STEP WAS REJECTED.\n"
            f"REASON: {invalid_reason}\n"
            f"INVALID STEP:\n{invalid_step}\n\n"
            "Rewrite only this step. Keep the same planned objective, but produce NEW, "
            "non-redundant, strict pro-claim content.\n"
            "RESPONSE FORMAT:\n"
            "STEP: [italian atomic step]"
        )

    def _validate_support_step_candidate(
        self, candidate_step: str, previous_steps: list[str], claim: str
    ) -> tuple[bool, str]:
        """Validation checks for one support step candidate."""
        text = (candidate_step or "").strip()
        if not text or text.upper() == "DONE":
            return False, "empty step"
        if self._is_garbage_text(text):
            return False, "garbage output"
        if not self._is_support_step_consistent(text):
            return False, "stance drift (not strictly pro-claim)"
        if not self._extract_cited_articles(text):
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
        prompt = f"""Valuta se il NUOVO passo aggiunge davvero informazione giuridica nuova.

CLAIM:
{claim}

RUOLO ARGOMENTATIVO: {role}

PASSI PRECEDENTI:
{context_prev}

NUOVO PASSO:
{candidate_step}

Regola:
- Rispondi REPEAT se il nuovo passo è sostanzialmente parafrasi/duplicazione dei precedenti.
- Rispondi NEW se introduce un punto giuridico/fattuale realmente diverso.

Rispondi con UNA SOLA parola: NEW oppure REPEAT.
"""
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

    def _evaluate_should_continue(
        self,
        claim: str,
        domain: str,
        steps: list[str],
        used_norms: list[str],
        knowledge_base: str,
        statutes_list: str,
        role: str = "support",
    ) -> bool:
        """Dedicated evaluation call: should the chain continue?

        A lightweight LLM call that inspects the current chain and
        decides whether an additional step would meaningfully
        strengthen the argument.

        Returns ``True`` to continue, ``False`` to conclude.
        """
        prev_context = "\n".join(
            f"  Step {i + 1}: {s[:300]}..." for i, s in enumerate(steps)
        )
        used_text = ", ".join(sorted(set(used_norms))) if used_norms else "none"
        n_steps = len(steps)
        n_unique_norms = len(set(used_norms)) if used_norms else 0

        role_desc = "supporting" if role == "support" else "counter-"

        eval_prompt = f"""You are a senior Italian jurist evaluating whether a legal argument needs more steps.

A {role_desc}argument for the claim below has {n_steps} steps so far.
It cites {n_unique_norms} unique norms out of {len(used_norms)} total citations.

CLAIM: "{claim}"
DOMAIN: {domain}

Steps so far:
{prev_context}

Norms already cited: {used_text}

AVAILABLE STATUTES (not yet used):
{statutes_list}

EVALUATION GUIDELINES:
- A well-constructed legal argument typically needs 3-6 distinct steps covering
  different legal aspects or applying norms to different facts.
- With only {n_steps} step(s), consider whether there are still pertinent aspects to cover.
- Each step should add NEW reasoning: a new norm, or a new application of an existing norm to a different fact.

CRITICAL — REPETITION DETECTION:
- REPETITION = two steps make the SAME legal point about the SAME factual aspect. This is BAD.
- GOOD COVERAGE = each step addresses a DIFFERENT legal aspect or applies a norm to a DIFFERENT fact.
- Re-citing an article is OK if applied to a genuinely different aspect of the case.
- If the last step simply rephrases or restates what a previous step already said, answer CONCLUDE.

DECISION RULES:
- Answer CONTINUE if ALL of these are true:
  (1) each step so far addresses a DIFFERENT legal aspect or fact (no repetition), AND
  (2) there is at least one pertinent unused norm OR a new factual aspect to cover, AND
  (3) fewer than 6 steps have been generated.
- Answer CONCLUDE if ANY of these is true:
  (a) any two steps make the SAME legal point about the SAME fact (repetition), OR
  (b) all pertinent legal aspects have been covered, OR
  (c) 6+ steps have already been generated.

YOUR ANSWER (exactly one word — CONTINUE or CONCLUDE):"""

        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=eval_prompt)])
            answer = (resp.content or "").strip().upper()
        except Exception as e:
            self._log(
                f"⚠️ Evaluation call failed: {e}; defaulting to CONTINUE",
                "warning",
            )
            return True

        # Robust parsing
        if "CONCLUD" in answer:
            should_continue = False
        elif "CONTINU" in answer:
            should_continue = True
        else:
            # Ambiguous → default to CONCLUDE after enough steps
            should_continue = n_steps < 5
            self._log(
                f"⚠️ Evaluator ambiguous: '{answer[:50]}'; "
                f"defaulting to {'CONTINUE' if should_continue else 'CONCLUDE'}",
                "warning",
            )

        self._log(f"🔍 Evaluator: {'CONTINUE' if should_continue else 'CONCLUDE'}")
        return should_continue

    def _build_support_stance_rewrite_prompt(
        self, original_prompt: str, invalid_step: str
    ) -> str:
        """Rewrite step text that drifted away from pro-claim stance."""
        return (
            f"{original_prompt}\n\n"
            "YOUR PREVIOUS STEP WAS INVALID because it weakens or contradicts the claim.\n"
            f'INVALID STEP:\n"{invalid_step}"\n\n'
            "Rewrite the SAME legal point with STRICT pro-claim stance.\n"
            "Do not add new facts. Do not balance pros and cons.\n\n"
            "RESPONSE FORMAT:\n"
            "STEP: [Italian text, max 4 sentences, strictly pro-claim]"
        )

    def _build_support_conclusion_rewrite_prompt(
        self, claim: str, chain_text: str, norms_text: str, invalid_conclusion: str
    ) -> str:
        """Force a concise conclusion aligned with pro-claim stance."""
        return f"""You are an expert Italian jurist. Rewrite the conclusion below so it STRICTLY supports the claim.

CLAIM:
"{claim}"

REASONING CHAIN:
{chain_text}

CITED NORMS: {norms_text}

INVALID CONCLUSION TO REWRITE:
"{invalid_conclusion}"

RULES:
- Keep it in Italian.
- 2-4 sentences max.
- Must clearly state the claim is legally founded.
- Must not include language suggesting rigetto/infondatezza/non annullabilita/legittimita dell'atto.
- Do not add facts or norms outside the chain.

CONCLUSION:"""

    def _is_support_step_consistent(self, step_text: str) -> bool:
        """
        Heuristic guardrail against anti-claim drift in support generation.

        Returns False when the text contains strong anti-claim signals
        not offset by explicit pro-claim language.
        """
        text = re.sub(r"\s+", " ", (step_text or "").strip().lower())
        if not text:
            return False

        pro_patterns = [
            r"\bpretesa\b.*\bfondat",
            r"\bricorso\b.*\bfondat",
            r"\bdeve\s+essere\s+accolt",
            r"\baccoglibil",
            r"\bannullabil",
            r"\b(il|lo)\s+atto\b.*\billegittim",
            r"\bprovvedimento\b.*\billegittim",
            r"\bviolazion",
            r"\bdetermina\b.*\billegittimit",
        ]
        anti_patterns = [
            r"\bpretesa\b.*\brigettat",
            r"\bricorso\b.*\brigettat",
            r"\binfondat",
            r"\binammissibil",
            r"\bnon\s+(?:e|è)?\s*annullabil",
            r"\bnon\s+determina\b.*\billegittimit",
            r"\b(?:atto|provvedimento)\b.{0,25}\blegittim",
        ]

        pro_score = 0
        for p in pro_patterns:
            if not re.search(p, text):
                continue
            if p == r"\bannullabil" and re.search(
                r"\bnon\s+(?:e|è)?\s*annullabil", text
            ):
                continue
            if p in (
                r"\b(il|lo)\s+atto\b.*\billegittim",
                r"\bprovvedimento\b.*\billegittim",
                r"\bdetermina\b.*\billegittimit",
            ) and re.search(
                r"\bnon\s+(?:rende|determina)\b.*\billegittim",
                text,
            ):
                continue
            if p == r"\bviolazion" and re.search(r"\bnon\b.{0,12}\bviolazion", text):
                continue
            if p == r"\bpretesa\b.*\bfondat" and re.search(
                r"\bnon\b.{0,12}\bfondat", text
            ):
                continue
            pro_score += 1

        anti_score = 0
        for p in anti_patterns:
            if not re.search(p, text):
                continue
            if p == r"\b(?:atto|provvedimento)\b.{0,25}\blegittim" and re.search(
                r"\bnon\b.{0,15}\blegittim", text
            ):
                continue
            anti_score += 1

        if re.search(r"\baccolt", text) and re.search(r"\brigettat", text):
            return False

        if anti_score >= 2 and pro_score == 0:
            return False
        if anti_score > pro_score + 1:
            return False
        return True

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

        prompt = f"""You are an expert Italian jurist. Based on the legal reasoning chain below, generate a concise and precise CONCLUSION.

ORIGINAL CLAIM:
"{claim}"

REASONING CHAIN:
{chain_text}

CITED NORMS: {norms_text}

INSTRUCTIONS:
- Write a conclusion of 2-4 sentences in Italian.
- The conclusion must SYNTHESIZE the result of the legal analysis, not repeat the individual steps.
- Clearly state whether the claim is legally founded or not and WHY, based on the norms analyzed.
- Do NOT introduce norms or facts not mentioned in the reasoning chain.
- Be direct and assertive in the final verdict.
- Your ENTIRE response must be written in Italian.

        CONCLUSION:"""

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
                        "⚠️ LLM conclusion drifts from pro-claim stance; rewriting",
                        "warning",
                    )
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
                        if rewritten and self._is_support_step_consistent(rewritten):
                            conclusion = rewritten
                        else:
                            self._log(
                                "⚠️ Rewritten conclusion still inconsistent; using fallback",
                                "warning",
                            )
                            conclusion = ""
                    except Exception as rewrite_exc:
                        self._log(
                            f"⚠️ Conclusion rewrite failed: {rewrite_exc}",
                            "warning",
                        )
                        conclusion = ""
                if conclusion:
                    self._log(f"📝 LLM-generated conclusion: {conclusion[:120]}...")
                    return conclusion
        except Exception as e:
            self._log(f"⚠️ Conclusion generation failed: {e}", "warning")

        # Static fallback
        return (
            f"Sulla base dell'analisi giuridica svolta, la pretesa risulta fondata. "
            f"Le norme richiamate ({norms_text}) trovano applicazione al caso di specie "
            f"e supportano il fondamento giuridico della domanda."
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

        if conclusion_text and not self._is_support_step_consistent(conclusion_text):
            self._log(
                "⚠️ Provided conclusion inconsistent with pro-claim stance; using fallback",
                "warning",
            )
            conclusion_text = ""

        if not conclusion_text:
            norms_list = ", ".join(norms) if norms else "le norme applicabili"
            conclusion_text = (
                f"Sulla base dell'analisi giuridica svolta, la pretesa risulta fondata. "
                f"Le norme richiamate ({norms_list}) trovano applicazione al caso di specie "
                f"e supportano il fondamento giuridico della domanda."
            )

        raw = (
            f"**Premessa**: {premise_text}\n\n"
            f"**Norma**:\n{norms_text}\n\n"
            f"**Nesso Causale**: La connessione tra le norme citate e la pretesa "
            f"emerge dalla catena di ragionamento sottostante, dove ciascun passo "
            f"costruisce logicamente sul precedente per dimostrare il fondamento "
            f"giuridico della domanda.\n\n"
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
        return f"""Analyze the following claim and build SUPPORTING arguments.

CLAIM:
"{claim}"

DOMAIN (from router):
{routing_decision.domain}

ANCHOR NORMS (structural constraints):
{anchor_text}

PRINCIPLE TESTS (evaluation criteria):
{principle_text}

=== KNOWLEDGE BASE (USE ONLY THESE SOURCES) ===
{knowledge_base}
=== END KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
{statutes_list}

ALLOWED PRECEDENT REFERENCES (do not cite others):
{precedents_list}

INSTRUCTIONS:
1) Build arguments appropriate for the {routing_decision.domain} domain.
2) Use anchor norms and principle tests as structural constraints, but DO NOT limit yourself to them.
   Your reasoning MUST cite multiple statutes from the KNOWLEDGE BASE — not only anchor norms.
   Anchor norms provide the framework, but you MUST integrate additional non-anchor statutes
   from the ALLOWED STATUTES list that are relevant to the specific facts of the claim.
   A good legal argument combines the general principle (anchor) with specific rules that apply
   to the concrete case (e.g., warranty, defects, remedies, damages, obligations).
3) If the knowledge base lacks a statute's text, still cite the article but do NOT invent quotes.
4) Build arguments using ONLY knowledge base sources, with EXACTLY these Italian headers:
   **Premessa**: (premise — write in prose, NO numbered lists)
   **Norma**: (statute with precise citation from ALLOWED STATUTES; if absent, write "articolo non disponibile nella knowledge base" — use bullet points with "-" if listing multiple norms, NEVER numbered lists)
   **Precedente**: (only if present in ALLOWED PRECEDENTS; otherwise omit — NO numbered lists)
   **Nesso Causale**: (causal link — write in prose, NO numbered lists)
   **Conclusione**: (conclusion — write in prose, NO numbered lists)
5) After the arguments, you MUST add the following header and numbered chain.
   This section is MANDATORY and must NEVER be omitted:

   **Catena di ragionamento**:
   1. [First reasoning step — cite the specific article(s) it relies on, e.g. Art. XX c.p.]
   2. [Second reasoning step — cite the specific article(s)]
   3. [Continue for each logical step...]

   RULES for the numbered chain:
   - Use EXACTLY the header "**Catena di ragionamento**:" before the numbered list.
   - Each step MUST be on its own line, starting with "N. " (e.g. "1. ", "2. ", "3. ").
   - Each step MUST reference at least one specific article (e.g. "Art. 2043 c.c.").
   - The chain must have AT LEAST 3 numbered steps.

FORMATTING RULE — CRITICAL:
- Numbered lists ("1. ", "2. ", "3. ", etc.) are ONLY allowed inside the **Catena di ragionamento** section.
- In ALL other sections (Premessa, Norma, Precedente, Nesso Causale, Conclusione), use ONLY
  prose text or bullet points with "-". NEVER use numbered lists outside the chain.

IMPORTANT - NORM USAGE REQUIREMENTS:
- You have {len(allowed_statutes)} statutes available. Cite EVERY article you deem pertinent
  to the case — do not artificially limit yourself to a fixed number.
- Do NOT rely on a single anchor norm for the entire chain.
- For each factual aspect of the claim (contract formation, defects, remedies, damages, etc.),
  identify the most specific applicable statute from the ALLOWED STATUTES list.
- Quote the relevant text from each statute when available in the KNOWLEDGE BASE.
- COHERENCE RULE: Every norm you cite in the **Norma** section MUST appear in at least one
  step of the numbered reasoning chain, with an explanation of its specific role in the argument.
  Do NOT list norms in **Norma** that you never use in the chain.

CRITICAL: Do not introduce external sources.
MANDATORY LANGUAGE RULE: Your ENTIRE response MUST be written in Italian. Do NOT write in English. Every sentence, header, and explanation must be in Italian."""

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
