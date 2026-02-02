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
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_statute_by_article_tool
from .tools.taxonomy_tools import get_causality_theory_tool

# System prompt for the Reasoner (with pre-retrieved context)
REASONER_SYSTEM_PROMPT = """You are the Reasoner. The router already set causal_type_id and theory_id.
Do NOT re-classify. Use these as structural constraints:
- anchor_norms (core + accessory) from config
- principle_tests for the causal type

You receive a pre-retrieved KNOWLEDGE BASE (statutes/precedents) filtered as supportive/neutral.
Build ONLY supporting arguments for the claim using the provided sources.

Critical rules:
- Cite ONLY statutes and precedents present in the KNOWLEDGE BASE.
- If a needed statute is missing, state “article not available in the knowledge base”.
- Keep reasoning independent: do not reference the Counter-Reasoner.
- Respond in Italian."""


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

    def run(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        pre_retrieved_statutes: list[dict],
        pre_retrieved_precedents: list[dict],
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

        # Phase 1: initial reasoning (no anchor injection)
        base_statutes = self._expand_with_cross_references(pre_retrieved_statutes)
        kb1 = self._format_context_for_prompt(base_statutes, pre_retrieved_precedents)
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
        domain = routing_decision.domain
        self._log(f"🔬 Router domain: {domain}")

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

        input_prompt = self._build_reasoning_prompt_with_context(
            claim,
            routing_decision,
            anchor_text,
            principle_text,
            knowledge_base,
            allowed_statutes,
            allowed_precedents,
        )

        raw_output, _ = self._invoke_reasoner(input_prompt)

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
        )

        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.arguments = self._extract_arguments(raw_output)
        output.reasoning_chain = self._sanitize_reasoning_chain(
            output.reasoning_chain, pre_retrieved_precedents
        )

        self._log(f"✅ Generated {len(output.arguments)} arguments", "success")
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke_reasoner(self, prompt: str) -> tuple[str, list]:
        """Invoke the ReAct agent and return (raw_output, messages)."""
        messages = [HumanMessage(content=prompt)]
        result = self.react_agent.invoke({"messages": messages})
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
            resp = self.llm.invoke([HumanMessage(content=prompt)])
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

    def filter_irrelevant_statutes(
        self, claim: str, statutes: list[dict]
    ) -> list[dict]:
        """
        Filter statutes using LLM one by one.
        Only discard when clearly unrelated; default to keeping on ambiguity.

        This is a PUBLIC method that can be called from api_server for pre-filtering.
        """
        if not statutes:
            self._log("No statutes to filter", "info")
            return statutes

        self._log(f"🔍 Filtering relevance: {len(statutes)} statutes initially")

        relevant_statutes = []

        for idx, statute in enumerate(statutes, start=1):
            article_number = statute.get("articolo", "N/A")
            article_title = statute.get("titolo", "Untitled")
            article_desc = statute.get("testo", "Untitled")

            prompt = f"""Legal Claim:
"{claim}"

Article:
"{article_number} - {article_title} - {article_desc}"

Instruction:
Determine whether the main topic of the article is directly mentioned or implied in the claim.

Rules:
- Do NOT evaluate whether the article fully resolves the issue.
- Do NOT suggest any additional articles.
- Do NOT use external knowledge; only consider the claim and this article.
- Do NOT add explanations or comments.
- Answer YES in all cases with even indirect connection.
- Use NO only when the article is clearly about a different domain.
- If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.
No punctuation. No new lines. No extra spaces.
"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                answer = response.content.strip().upper()
            except Exception as e:
                self._log(
                    f"⚠️ LLM call failed for article {article_number}: {e}", "warning"
                )
                answer = "YES"

            token = answer.split()[0] if answer else ""
            keep = token != "NO" and (
                token == "YES" or "YES" in answer or "NO" not in answer
            )

            if keep:
                relevant_statutes.append(statute)
                self._log(
                    f"✅ Keeping article [{idx}] {article_number} - {article_title}"
                )
            else:
                self._log(
                    f"❌ Discarding article [{idx}] {article_number} - {article_title}",
                    "warning",
                )

        self._log(f"📊 Result: {len(relevant_statutes)}/{len(statutes)} statutes kept")
        return relevant_statutes

    def filter_irrelevant_precedents(
        self, claim: str, precedents: list[dict]
    ) -> list[dict]:
        """
        Soft-filter precedents: keep by default, discard only when clearly unrelated.
        """
        if not precedents:
            self._log("No precedents to filter", "info")
            return precedents

        self._log(f"🔍 Filtering relevance: {len(precedents)} precedents initially")

        relevant_precedents = []

        for idx, precedent in enumerate(precedents, start=1):
            title = precedent.get("title", "Untitled")
            summary = precedent.get("summary", "")

            prompt = f"""Legal Claim:
"{claim}"

Precedent:
"{title}" - "{summary}"

Instruction:
Decide if this precedent has a meaningful connection to the claim (employment/licenziamento, retribuzione, TFR, rapporto di lavoro).

Rules:
- Answer YES unless the precedent is clearly about a different domain (es. societario puro, titoli di credito, marchi, appalti pubblici) with no employment link.
- If there is any plausible link to employment/termination/worker rights, answer YES.
- If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.
No punctuation. No new lines. No extra spaces.
"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                answer = response.content.strip().upper()
            except Exception as e:
                self._log(
                    f"⚠️ LLM call failed for precedent [{idx}] {title}: {e}", "warning"
                )
                answer = "YES"

            token = answer.split()[0] if answer else ""
            keep = token != "NO" and (
                token == "YES" or "YES" in answer or "NO" not in answer
            )

            if keep:
                relevant_precedents.append(precedent)
                self._log(f"✅ Keeping precedent [{idx}] {title}")
            else:
                self._log(f"❌ Discarding precedent [{idx}] {title}", "warning")

        self._log(
            f"📊 Result: {len(relevant_precedents)}/{len(precedents)} precedents kept"
        )
        return relevant_precedents

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
2) Use anchor norms and principle tests as constraints: if the knowledge base lacks the statute text, still cite the article but do NOT invent quotes.
3) Build arguments using ONLY knowledge base sources:
   - Premise
   - Statute (with precise citation) from ALLOWED STATUTES; if absent, write “article not available in the knowledge base”
   - Precedent (only if present in ALLOWED PRECEDENTS)
   - Causal Link
   - Conclusion
4) End with a numbered reasoning chain that respects anchor norms and principle tests.

Critical: do not introduce external sources. Respond in Italian."""

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

    def _anchor_norms_to_statutes(self, anchor_norms: dict) -> list[dict]:
        """Convert anchor norms into statute-like dicts for prompt allow-list."""
        if not anchor_norms:
            return []
        combined = anchor_norms.get("core_norms", []) + anchor_norms.get(
            "accessory_norms", []
        )
        result = []
        for n in combined:
            # Support both old keys (riferimento/nota) and new keys (ref/role)
            ref = n.get("ref") or n.get("riferimento", "")
            role = n.get("role") or n.get("nota", "")
            if ref:
                result.append(
                    self._norm_to_statute_dict({"riferimento": ref, "nota": role})
                )
        return result

    def _expand_with_cross_references(self, statutes: list[dict]) -> list[dict]:
        """
        Add statutes explicitly referenced inside the text of already provided articles.
        Useful when an article (e.g., 2056 c.c.) rinvia ad altri (1223/1226/1227).
        """
        try:
            import re
        except Exception:
            return statutes

        seen = {(s.get("articolo"), s.get("source")) for s in statutes}
        extra: list[dict] = []

        pattern = re.compile(r"art\.?\s*(\d{2,4})", re.IGNORECASE)

        for s in statutes:
            text = s.get("testo") or ""
            refs = set(pattern.findall(text))

            # Also catch slash-separated numbers like "1223/1226/1227"
            for token in re.findall(r"\b(\d{2,4})\b", text):
                if "/" in token:
                    continue
                refs.add(token)

            added_refs: list[str] = []
            for ref in refs:
                key = (ref, s.get("source"))
                if key in seen:
                    continue
                seen.add(key)
                result = get_statute_by_article_tool.invoke(
                    {"articolo": ref, "codice": s.get("source", "codice_civile")}
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
