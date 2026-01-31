"""
LexCausa Reasoner Agent.

The Reasoner is responsible for:
1. Receiving a legal claim with pre-retrieved statutes and precedents
2. Generating supporting arguments based on the provided knowledge base
3. Building a reasoning chain that connects the claim to legal norms

NOTE: The Reasoner does NOT classify causality. Causality classification
is performed AFTER the reasoning chain is generated, using the chain itself
as input (not the claim). This classification is then used by the CounterReasoner.

IMPORTANT: The Reasoner does NOT search for articles/precedents itself.
The pre-retrieval is done by api_server using LegalSearchPipeline.

Uses LangGraph with Groq Cloud for LLM-powered reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import get_statute_by_article_tool

# System prompt for the Reasoner (NO causality classification)
REASONER_SYSTEM_PROMPT = """You are an expert legal reasoning agent specializing in Italian law.

You will receive a legal claim along with PRE-RETRIEVED articles and precedents as your KNOWLEDGE BASE.
Your task is to analyze the claim and build supporting arguments using ONLY the provided knowledge.

CRITICAL RULES:
- Use ONLY the articles provided in the KNOWLEDGE BASE - do NOT invent or cite articles not provided
- Use ONLY the precedents provided - do NOT invent precedents
- If no precedents are provided, explicitly state this and proceed without them
- ALWAYS cite the exact article number and code (e.g., "Art. 2043 c.c.", "Art. 40 c.p.")
- ALWAYS quote relevant portions of the article text from the provided context

Your task follows these steps:

1. **ARGUMENT CONSTRUCTION**: For each supporting argument:
   - **Premessa**: The starting fact from the claim
   - **Norma**: The applicable law WITH EXACT CITATION and quoted text from knowledge base
   - **Precedente**: A supporting court decision (ONLY if provided in knowledge base)
   - **Nesso Causale**: How the premise connects to the norm
   - **Conclusione**: The legal implication

2. **REASONING CHAIN**: Build a logical sequence:
   Facts → Applicable Norm (with citation) → Precedent Support → Causal Link → Legal Consequence

The response language must be Italian."""


@dataclass
class ReasonerOutput:
    """Structured output from the Reasoner."""

    claim: str
    causality_classification: dict
    relevant_statutes: list[dict] = field(default_factory=list)
    relevant_precedents: list[dict] = field(default_factory=list)
    arguments: list[dict] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causality": self.causality_classification,
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

        NOTE: No search tools and no causality tools.
        The agent works with pre-retrieved context and generates reasoning without causality classification.
        Causality is classified AFTER on the generated reasoning chain.
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
        pre_retrieved_statutes: list[dict],
        pre_retrieved_precedents: list[dict],
    ) -> ReasonerOutput:
        """
        Execute the reasoning process on a legal claim with pre-retrieved knowledge.

        Args:
            claim: The legal claim to analyze and support.
            pre_retrieved_statutes: Already retrieved and filtered statute articles.
            pre_retrieved_precedents: Already retrieved precedents.

        Returns:
            ReasonerOutput with causality classification, sources, and arguments.
        """
        self._log(f"Analyzing claim: {claim[:100]}...")
        self._log(
            f"📚 Knowledge base: {len(pre_retrieved_statutes)} statutes, {len(pre_retrieved_precedents)} precedents"
        )

        if not pre_retrieved_statutes and not pre_retrieved_precedents:
            self._log("⚠️ No knowledge base provided", "warning")
            return ReasonerOutput(
                claim=claim,
                causality_classification={},  # Will be set by api_server after
                relevant_statutes=[],
                relevant_precedents=[],
                arguments=[],
                reasoning_chain=["Nessun articolo o precedente fornito per l'analisi."],
                raw_response="Analisi non completata: nessuna fonte normativa o giurisprudenziale disponibile.",
            )

        # Prepare statutes for the prompt (no taxonomy enrichment)
        deduped_statutes = pre_retrieved_statutes
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

        # Build the input prompt with pre-retrieved context and explicit allow-list
        input_prompt = self._build_reasoning_prompt_with_context(
            claim,
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
            if hasattr(msg, 'name') and msg.name:
                tool_names.append(msg.name)
                self._log(f"🔧 Tool called: {msg.name}")

        if tool_names:
            self._log(f"📊 Tools used: {', '.join(set(tool_names))}")

        # NOTE: Causality is NOT extracted here - it will be classified by api_server
        # using the reasoning chain as input (not the claim)

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

        # Build output (causality_classification is empty - set by api_server after)
        output = ReasonerOutput(
            claim=claim,
            causality_classification={},  # Will be populated by api_server
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
        knowledge_base: str,
        allowed_statutes: list[str],
        allowed_precedents: list[str],
    ) -> str:
        """Build the prompt for the reasoning task with pre-retrieved context."""
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes)
            or "- Nessun articolo disponibile"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- Nessun precedente disponibile"
        )
        return f"""Analyze the following legal claim and build supporting arguments.

CLAIM:
"{claim}"

=== KNOWLEDGE BASE (USE ONLY THESE SOURCES) ===
{knowledge_base}
=== END OF KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
{statutes_list}

ALLOWED PRECEDENT REFERENCES (do not cite others):
{precedents_list}

INSTRUCTIONS:

1. **BUILD ARGUMENTS**: Using ONLY the articles and precedents from the KNOWLEDGE BASE above:
   - For each argument, specify:
     - **Premessa**: The starting fact from the claim
     - **Norma**: Article citation with quoted text FROM THE KNOWLEDGE BASE (only from ALLOWED STATUTE REFERENCES)
     - **Precedente**: Supporting precedent (ONLY if present in ALLOWED PRECEDENTS; if none, OMIT this field)
      - **Nesso Causale**: How the premise connects to the norm
      - **Conclusione**: The legal implication

2. **REASONING CHAIN**: Provide a final CATENA DI RAGIONAMENTO with:
   - Numbered logical steps
   - Explicit article citations (e.g., "Art. 2043 c.c.")
   - Precedent references if available

CRITICAL:
- Cite ONLY the statutes listed in ALLOWED STATUTE REFERENCES. If a needed article is missing, write "articolo non disponibile nel knowledge base".
- Cite ONLY the precedents in ALLOWED PRECEDENTS; if none apply, omit the precedents field rather than stating it is unavailable.

The response language must be Italian."""

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
