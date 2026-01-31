"""
LexCausa Reasoner Agent.

The Reasoner is responsible for:
1. Receiving a legal claim with pre-retrieved statutes and precedents
2. Using ReAct logic with tools to classify causality and get theory
3. Generating supporting arguments based on the provided knowledge base
4. Building a reasoning chain that connects the claim to legal norms

IMPORTANT: The Reasoner does NOT search for articles/precedents itself.
The pre-retrieval is done by api_server using LegalSearchPipeline.
This ensures the agent bases its reasoning ONLY on the retrieved knowledge.

Uses LangGraph with Groq Cloud for LLM-powered reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .router import RoutingDecision
from .tools.neo4j_tools import get_statute_by_article_tool

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
- Respond in English."""


@dataclass
class ReasonerOutput:
    """Structured output from the Reasoner."""

    claim: str
    causality_classification: dict
    causal_type_id: str = ""
    theory_id: str = ""
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
        Execute the reasoning process on a legal claim with pre-retrieved knowledge.

        Args:
            claim: The legal claim to analyze and support.
            routing_decision: Output of the Router containing causal_type_id/theory_id.
            pre_retrieved_statutes: Already retrieved and filtered statute articles.
            pre_retrieved_precedents: Already retrieved precedents.

        Returns:
            ReasonerOutput with causality classification, sources, and arguments.
        """
        self._log(f"Analyzing claim: {claim[:100]}...")
        self._log(
            f"📚 Knowledge base: {len(pre_retrieved_statutes)} statutes, {len(pre_retrieved_precedents)} precedents"
        )

        if not routing_decision or not routing_decision.causal_type_id:
            raise ValueError("routing_decision with a valid causal_type_id is required")

        if not pre_retrieved_statutes and not pre_retrieved_precedents:
            self._log("⚠️ No knowledge base provided", "warning")
            return ReasonerOutput(
                claim=claim,
                causality_classification={
                    "causal_type_id": routing_decision.causal_type_id,
                    "theory_id": routing_decision.theory_id,
                    "source": "router",
                },
                causal_type_id=routing_decision.causal_type_id,
                theory_id=routing_decision.theory_id,
                anchor_norms=routing_decision.anchor_norms,
                principle_tests=routing_decision.principle_tests,
                relevant_statutes=[],
                relevant_precedents=[],
                arguments=[],
                reasoning_chain=[
                    "No statutes or precedents were provided for analysis."
                ],
                raw_response="Analysis not completed: no statutory or case sources available.",
            )

        anchor_statutes = self._anchor_norms_to_statutes(routing_decision.anchor_norms)
        if anchor_statutes:
            added_refs = [
                str(s.get("articolo")) for s in anchor_statutes if s.get("articolo")
            ]
            self._log(
                f"🧭 Anchor norms added to KB: {len(anchor_statutes)} "
                f"({', '.join(added_refs)})"
            )

        # Merge and deduplicate statutes coming from retrieval (already supportive/neutral)
        all_statutes = pre_retrieved_statutes + anchor_statutes
        seen_keys = set()
        deduped_statutes = []
        for s in all_statutes:
            key = (s.get("articolo"), s.get("source"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_statutes.append(s)

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

        # Format the knowledge base for the prompt
        knowledge_base = self._format_context_for_prompt(
            deduped_statutes, pre_retrieved_precedents
        )

        anchor_text = self._format_anchor_norms(routing_decision.anchor_norms)
        principle_text = self._format_principle_tests(routing_decision.principle_tests)

        # Build the input prompt with pre-retrieved context and explicit allow-list
        input_prompt = self._build_reasoning_prompt_with_context(
            claim,
            routing_decision,
            anchor_text,
            principle_text,
            knowledge_base,
            allowed_statutes,
            allowed_precedents,
        )

        # Execute the ReAct agent using LangGraph
        messages = [HumanMessage(content=input_prompt)]
        result = self.react_agent.invoke({"messages": messages})
        messages_out = result.get("messages", [])

        # Log tool calls for debugging
        tool_names: list[str] = []
        for msg in messages_out:
            if hasattr(msg, "name") and msg.name:
                tool_names.append(msg.name)

        if tool_names:
            self._log(f"📊 Tools used: {', '.join(set(tool_names))}")

        # Extract causality from tool responses
        causality = {
            "causal_type_id": routing_decision.causal_type_id,
            "theory_id": routing_decision.theory_id,
            "source": "router",
        }

        # Get the final response from the agent
        raw_output = ""
        for msg in reversed(messages_out):
            # Skip tool responses; keep the final LLM message
            if isinstance(msg, ToolMessage):
                continue
            msg_content = getattr(msg, "content", None)
            if msg_content:
                raw_output = str(msg_content)
                break

        # Build output
        output = ReasonerOutput(
            claim=claim,
            causality_classification=causality,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            anchor_norms=routing_decision.anchor_norms,
            principle_tests=routing_decision.principle_tests,
            relevant_statutes=deduped_statutes,
            relevant_precedents=pre_retrieved_precedents,
            raw_response=raw_output,
        )

        # Parse the response for structured data
        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.arguments = self._extract_arguments(raw_output)

        # Sanitize reasoning chain based on precedents
        output.reasoning_chain = self._sanitize_reasoning_chain(
            output.reasoning_chain, pre_retrieved_precedents
        )

        self._log(f"✅ Generated {len(output.arguments)} arguments", "success")
        return output

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
        return f"""Analyze the following claim and build SUPPORTING arguments following the router decision.

CLAIM:
"{claim}"

ROUTING DECISION (binding):
- causal_type_id: {routing_decision.causal_type_id}
- theory_id: {routing_decision.theory_id}

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
1) Honor the given causal_type_id and theory_id. Do not re-classify.
2) Use anchor norms and principle tests as constraints: if the knowledge base lacks the statute text, still cite the article but do NOT invent quotes.
3) Build arguments using ONLY knowledge base sources:
   - Premise
   - Statute (with precise citation) from ALLOWED STATUTES; if absent, write “article not available in the knowledge base”
   - Precedent (only if present in ALLOWED PRECEDENTS)
   - Causal Link
   - Conclusion
4) End with a numbered reasoning chain that respects anchor norms and principle tests.

Critical: do not introduce external sources. Respond in English."""

    def _format_anchor_norms(self, anchor_norms: dict) -> str:
        """Format anchor norms for prompt readability."""
        core = anchor_norms.get("core_norms", []) if anchor_norms else []
        accessory = anchor_norms.get("accessory_norms", []) if anchor_norms else []
        lines = []
        for n in core:
            lines.append(f"- [core] {n.get('ref', 'N/D')}: {n.get('role', '')}")
        for n in accessory:
            lines.append(f"- [accessory] {n.get('ref', 'N/D')}: {n.get('role', '')}")
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
        return [
            self._norm_to_statute_dict(
                {"riferimento": n.get("ref", ""), "nota": n.get("role", "")}
            )
            for n in combined
        ]

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
