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
                f"Art. {s.get('articolo')} ({'c.c.' if s.get('source') == 'codice_civile' else 'c.p.'})"
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
            f"Art. {s.get('articolo')} ({'c.c.' if s.get('source') == 'codice_civile' else 'c.p.'})"
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
        # Pattern per articoli: "Art. 2043 c.c.", "art. 1223 c.p.", "articolo 40 c.p.", etc.
        pattern = re.compile(
            r"(?:art(?:icolo)?\.?\s*)(\d{1,4})\s*(c\.?[cp]\.?|cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?))",
            re.IGNORECASE,
        )
        matches = pattern.findall(text)
        articles = []
        for num, code in matches:
            code_norm = (
                "c.c." if "c" in code.lower() and "p" not in code.lower() else "c.p."
            )
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
            # "civile" or "penale"
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
        """Generate the reasoning chain step-by-step with dedicated LLM calls.

        Each iteration produces ONE substantive reasoning step.  The LLM
        receives full context of prior steps and autonomously decides
        whether to continue or conclude.  ``chain_max_steps`` acts only
        as a safety cap.

        Returns
        -------
        (raw_response, reasoning_chain)
            The assembled raw text and the list of individual step texts.
        """
        MAX_STEPS = settings.chain_max_steps
        MIN_STEPS = settings.chain_min_steps
        steps: list[str] = []
        used_norms: list[str] = []

        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )

        for step_num in range(1, MAX_STEPS + 1):
            is_last_possible = step_num == MAX_STEPS
            can_conclude = step_num >= MIN_STEPS

            # --- EVALUATION PHASE (separate LLM call) ---
            if can_conclude and not is_last_possible and steps:
                should_continue = self._evaluate_should_continue(
                    claim=claim,
                    domain=routing_decision.domain,
                    steps=steps,
                    used_norms=used_norms,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    role="support",
                )
                if not should_continue:
                    self._log(
                        f"🏁 Chain concluded at step {step_num - 1} "
                        f"by evaluator (before generating step {step_num})"
                    )
                    break

            # Build context of previous steps
            if steps:
                prev_context = "\n".join(
                    f"  Step {i + 1}: {s}" for i, s in enumerate(steps)
                )
                used_norms_text = ", ".join(used_norms) if used_norms else "none"
            else:
                prev_context = "  (No previous steps — you are at step 1.)"
                used_norms_text = "none"

            # ---- Per-step prompt (English, response in Italian) ----
            last_step_notice = (
                "\nTHIS IS THE LAST ALLOWED STEP. "
                "You MUST conclude the argument now."
                if is_last_possible
                else ""
            )
            step_prompt = f"""You are an expert Italian jurist. You are building a SUPPORTING argument STEP BY STEP for the following legal claim.

YOUR STANCE: You are the ADVOCATE of the claim. You MUST argue that the claim IS legally founded.
Every step must provide ONE concrete reason WHY the claim succeeds under Italian law.
NEVER mention weaknesses, counter-arguments, possible objections, or doubts about the claim.
Do NOT balance pros and cons. You are EXCLUSIVELY pro-claim.

CLAIM (you must SUPPORT this):
"{claim}"

DOMAIN: {routing_decision.domain}

ANCHOR NORMS (structural constraints):
{anchor_text}

PRINCIPLE TESTS (evaluation criteria):
{principle_text}

=== KNOWLEDGE BASE (use ONLY these sources) ===
{knowledge_base}
=== END KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
{statutes_list}

ALLOWED PRECEDENT REFERENCES (do not cite others):
{precedents_list}

--- CURRENT ARGUMENT STATE ---
Steps completed so far:
{prev_context}

Norms already used: {used_norms_text}
Current step: {step_num} (safety cap: {MAX_STEPS})

--- INSTRUCTIONS FOR STEP {step_num} ---
Generate EXACTLY ONE ATOMIC reasoning step (step {step_num}).

ATOMIC STEP RULES:
- This step is ONE SMALL PIECE of a multi-step logical chain. Do NOT try to give a complete answer.
- Focus on EXACTLY ONE legal point, norm, or factual aspect. Do NOT cover multiple aspects.
- 2-4 sentences MAXIMUM. Be concise and precise.
- Do NOT repeat or summarize the claim. The claim is already known.
- Do NOT restate conclusions from previous steps. Build on them.

PRO-CLAIM REQUIREMENTS for this step:
1. ONE SUPPORTING POINT ONLY: Pick exactly ONE of the following for this step:
   - Show how ONE specific norm SUPPORTS the claim on ONE specific fact, OR
   - Demonstrate that ONE legal prerequisite IS satisfied, OR
   - Draw ONE narrow conclusion showing the claim IS legally grounded
2. NORM COVERAGE: Try to cite an article NOT yet used ({used_norms_text}).
   You MAY reuse an already-cited article ONLY if you apply it to a DIFFERENT factual aspect
   that was NOT discussed in any previous step. Never repeat the same reasoning.
   If you have nothing new to add (no new aspect, no new norm), respond with STEP: DONE.
3. PRECEDENT CITATION: If a precedent from the ALLOWED PRECEDENT REFERENCES list directly
   supports your reasoning point, you MUST cite it by including its FULL EXACT TITLE in the step text.
   For example: "Come confermato dalla giurisprudenza in «Titolo completo del precedente», ..."
   Do NOT rephrase or shorten the title — copy it exactly as listed.
   If no precedent is relevant for this step, skip this and cite only the norm.
4. CONNECT to the previous step: your step must start from where the last step ended.
   If step N-1 established X, step N should use X to advance to Y.
5. ALWAYS FAVOR THE CLAIM: interpret norms and facts in the way most favorable to the claimant.
{last_step_notice}

RESPONSE FORMAT:
STEP: [Your atomic reasoning step in Italian — max 4 sentences]

CRITICAL RULES:
- Your ENTIRE STEP text must be written in Italian.
- MAX 4 sentences. If you need more, you are covering too much — split it.
- Cite exactly one specific article (e.g. Art. 2043 c.c.) and, when relevant, one precedent by its FULL EXACT TITLE from the ALLOWED PRECEDENT REFERENCES list.
- FACTUAL FIDELITY: Use ONLY facts explicitly stated in the CLAIM above. Do NOT add, infer, assume, or invent facts that are not written in the claim. If the claim says the person struck once, do not say they struck multiple times.
- Do NOT invent sources not present in the knowledge base.
- Do NOT write a complete argument. Write ONE building block.
- NEVER write anything that weakens or questions the claim. Every sentence must SUPPORT it.
"""

            self._log(f"🔗 Generating step {step_num}/{MAX_STEPS}...")

            try:
                resp = self._resilient_llm_invoke(
                    [HumanMessage(content=step_prompt)],
                    stream_callback=(
                        (
                            lambda token: self._emit_stream_token(
                                stream_callback,
                                phase="support",
                                token=token,
                                step=step_num,
                            )
                        )
                        if stream_callback
                        else None
                    ),
                )
                step_response = (resp.content or "").strip()
            except Exception as e:
                self._log(f"⚠️ Step {step_num} generation failed: {e}", "warning")
                break

            step_text = self._parse_step_text(step_response)

            if not step_text or step_text.strip().upper() == "DONE":
                self._log(
                    f"⚠️ Step {step_num}: no new norm available, stopping", "warning"
                )
                break

            # --- GARBAGE DETECTION (degenerate LLM output) ---
            if self._is_garbage_text(step_text):
                self._log(
                    f"🗑️ Step {step_num}: garbage/degenerate output detected "
                    f"(token repetition loop), discarding and stopping chain",
                    "warning",
                )
                break

            # --- REPETITION DETECTION (programmatic) ---
            if steps and self._is_repetitive_step(step_text, steps):
                self._log(
                    f"🔁 Step {step_num}: too similar to a previous step, "
                    f"stopping chain (repetition detected)"
                )
                break

            steps.append(step_text)

            new_norms = self._extract_cited_articles(step_text)
            used_norms.extend(new_norms)

            # Detect precedent mentions in step text
            prec_mentions = [
                p for p in allowed_precedents if p.lower() in step_text.lower()
            ]
            prec_info = f" | prec: {', '.join(prec_mentions)}" if prec_mentions else ""

            self._log(
                f"✅ Step {step_num}: {step_text[:80]}... "
                f"| norms: {', '.join(new_norms) if new_norms else 'none'}{prec_info}"
            )

            # Last possible step: forced stop
            if is_last_possible:
                self._log(f"🏁 Chain stopped at safety cap (step {step_num})")
                break

        if not steps:
            self._log("❌ No steps generated in iterative chain", "error")
            return "", []

        self._log(
            f"📊 Iterative chain complete: {len(steps)} steps, "
            f"{len(set(used_norms))} unique norms"
        )

        raw_response = self._assemble_raw_response(claim, steps)
        return raw_response, steps

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
