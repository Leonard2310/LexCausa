"""
LexCausa Reasoner Agent.

The Reasoner is responsible for:
1. Receiving a legal claim
2. Retrieving relevant statutes and precedents from Neo4j KB
3. Classifying the causality type (Materiale, Giuridica, Concause)
4. Generating supporting arguments based on the retrieved information
5. Building a reasoning chain that connects the claim to legal norms

Uses LangGraph with Groq Cloud for LLM-powered reasoning.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import (
    get_statute_by_article_tool,
    search_legal_sources_tool,
    search_precedents_tool,
)
from .tools.taxonomy_tools import classify_causality_tool, get_causality_theory_tool

# System prompt for the Reasoner
REASONER_SYSTEM_PROMPT = """You are an expert legal reasoning agent specializing in Italian law.

Your task is to analyze legal claims and build supporting arguments following these steps:

1. **CLAIM ANALYSIS**: Understand the legal question posed by the claim.

2. **CAUSALITY CLASSIFICATION**: Determine the relevant type of causality:
   - Material Causality: factual link between conduct and event (Art. 40 c.p.)
   - Legal Causality: connection between event and compensable damage (Art. 1223 c.c.)
   - Concurrent/Supervening Causes: interaction between multiple causal factors (Art. 41 c.p.)

3. **NORMATIVE RESEARCH**: Use the available tools to find:
   - Relevant law articles (Civil Code and/or Criminal Code)
   - Relevant jurisprudential precedents

4. **ARGUMENT CONSTRUCTION**: For each supporting argument:
   - Identify the factual premise
   - Connect to the applicable norm with EXPLICIT CITATION (e.g., "Art. 2043 c.c.")
   - Quote the relevant text from the article
   - Explain the causal link
   - Conclude with the legal implication

5. **PRECEDENT INTEGRATION**: For each precedent found:
   - Cite the precedent explicitly (court, date, case number if available)
   - Quote the relevant holding or principle
   - Explain how it supports your argument
   - Integrate it into the reasoning chain

6. **REASONING CHAIN**: Build a logical sequence that EXPLICITLY includes:
   Facts → Applicable Norm (with citation) → Precedent Support → Causal Link → Legal Consequence

CRITICAL RULES:
- ALWAYS cite the exact article number and code (e.g., "Art. 2043 c.c.", "Art. 40 c.p.")
- ALWAYS quote relevant portions of the article text
- ALWAYS cite precedents with their identifying information
- ALWAYS explain how precedents support your reasoning
- Precedents are MANDATORY in the final reasoning chain

Always respond in Italian and be precise with normative references."""


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
        }


class Reasoner(BaseAgent):
    """
    Legal Reasoner Agent.

    Analyzes legal claims, retrieves relevant legal sources from Neo4j,
    classifies causality type, and generates supporting arguments.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Reasoner agent."""
        super().__init__(config)
        self._react_agent = None

    @property
    def tools(self) -> list:
        """Get the tools available to this agent."""
        return [
            # PRIMARY: replicates Tab Ricerca exactly
            search_legal_sources_tool,
            # Secondary tools
            get_statute_by_article_tool,
            search_precedents_tool,
            classify_causality_tool,
            get_causality_theory_tool,
        ]

    @property
    def react_agent(self):
        """Lazy initialization of the ReAct agent using LangGraph."""
        if self._react_agent is None:
            # Create ReAct agent with LangGraph
            self._react_agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=REASONER_SYSTEM_PROMPT,
            )
        return self._react_agent

    def run(
        self,
        claim: str,
        include_precedents: bool = True,
        max_statutes: int = 5,
        max_precedents: int = 3,
    ) -> ReasonerOutput:
        """
        Execute the reasoning process on a legal claim.

        Args:
            claim: The legal claim to analyze and support.
            include_precedents: Whether to search for precedents.
            max_statutes: Maximum number of statutes to retrieve.
            max_precedents: Maximum number of precedents to retrieve.

        Returns:
            ReasonerOutput with causality classification, sources, and arguments.
        """
        self._log(f"Analyzing claim: {claim[:100]}...")

        # Build the input prompt
        input_prompt = self._build_reasoning_prompt(
            claim, include_precedents, max_statutes, max_precedents
        )

        # Execute the ReAct agent using LangGraph
        messages = [HumanMessage(content=input_prompt)]
        result = self.react_agent.invoke({"messages": messages})
        messages_out = result.get("messages", [])

        # Log tool calls for debugging
        tool_calls = []
        for msg in messages_out:
            if isinstance(msg, ToolMessage):
                tool_calls.append(msg.name)
                self._log(f"🔧 Tool called: {msg.name}")

        if tool_calls:
            self._log(f"📊 Tools used: {', '.join(tool_calls)}")
        else:
            self._log("⚠️ No tools were called by the agent")

        # Extract data from tool responses
        causality = self._extract_causality_from_messages(messages_out)
        statutes = self._extract_statutes_from_messages(messages_out)
        precedents = self._extract_precedents_from_messages(messages_out)

        if statutes:
            self._log(f"📜 Found {len(statutes)} statutes")
        if precedents:
            self._log(f"⚖️ Found {len(precedents)} precedents")

        # Extract final response from messages
        raw_output = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    raw_output = msg.content
                    break

        # Parse the response
        output = self._parse_response(claim, {"output": raw_output})
        output.causality_classification = causality
        output.relevant_statutes = statutes
        output.relevant_precedents = precedents

        self._log(f"Generated {len(output.arguments)} arguments", "success")
        return output

    def _extract_causality_from_messages(self, messages) -> dict:
        """
        Extract the result of the classify_causality tool from LangGraph messages.
        """
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "classify_causality":
                try:
                    return json.loads(msg.content)
                except Exception:
                    return {}
        return {}

    def _extract_statutes_from_messages(self, messages) -> list[dict]:
        """
        Extract statutes from search_legal_sources or search_statutes tool responses.
        Filters out error/empty messages.
        """
        statutes = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)

                    # Handle search_legal_sources (primary tool - new format)
                    if msg.name == "search_legal_sources":
                        if isinstance(data, dict) and "articles" in data:
                            for item in data["articles"]:
                                if isinstance(item, dict) and "statute_id" in item:
                                    statutes.append(item)

                    # Handle search_statutes (secondary tool - list format)
                    elif msg.name == "search_statutes":
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "statute_id" in item:
                                    statutes.append(item)
                except Exception:
                    pass
        return statutes

    def _extract_precedents_from_messages(self, messages) -> list[dict]:
        """
        Extract precedents retrieved by search_precedents tool.
        Filters out error/empty messages.
        """
        precedents = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_precedents":
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, list):
                        # Filter out error/empty messages
                        for item in data:
                            if isinstance(item, dict) and "precedent_id" in item:
                                precedents.append(item)
                except Exception:
                    pass
        return precedents

    def _build_reasoning_prompt(
        self,
        claim: str,
        include_precedents: bool,
        max_statutes: int,
        max_precedents: int,
    ) -> str:
        """Build the prompt for the reasoning task."""
        prompt = f"""Analyze the following legal claim and build supporting arguments.

CLAIM:
"{claim}"

INSTRUCTIONS (follow this order STRICTLY):

1. **SEARCH LEGAL SOURCES**: Call `search_legal_sources` with the COMPLETE claim text.
   CRITICAL: Pass the EXACT original claim above - do NOT rephrase or summarize it!
   This tool will automatically classify the claim and find relevant articles.
   Use: search_legal_sources(claim="{claim}", top_k={max_statutes})

2. **CLASSIFY CAUSALITY**: Call `classify_causality` to determine the type of causality.
"""

        if include_precedents:
            prompt += f"""
3. **SEARCH PRECEDENTS**: Call `search_precedents` with the claim text (max {max_precedents} results)

4. **GET CAUSAL THEORY**: Call `get_causality_theory` to retrieve the complete theory.

5. **BUILD ARGUMENTS**: Construct structured arguments based on the retrieved information.
"""
        else:
            prompt += """
3. **GET CAUSAL THEORY**: Call `get_causality_theory` to retrieve the complete theory.

4. **BUILD ARGUMENTS**: Construct structured arguments based on the retrieved information.
"""

        prompt += """
For each argument, you MUST specify:
- **Premessa**: The starting fact or situation from the claim
- **Norma**: The applicable law article with EXACT CITATION (e.g., "Art. 2043 c.c.") AND quote the relevant text
- **Precedente**: A supporting court decision - cite it and explain how it applies
- **Nesso Causale**: How the premise connects to the norm, supported by the precedent
- **Conclusione**: The legal implication

At the end, provide a CATENA DI RAGIONAMENTO that:
1. Lists each logical step
2. EXPLICITLY cites the articles used (with article number and code)
3. EXPLICITLY cites the precedents used (with title/reference)
4. Shows how precedents reinforce the legal reasoning

EXAMPLE FORMAT for citations:
- "Ai sensi dell'Art. 2043 c.c., che dispone: '[testo rilevante]'..."
- "Come stabilito dalla Corte di Cassazione in [riferimento]: '[massima]'..."

Respond in structured format and in Italian."""

        return prompt

    def _parse_response(self, claim: str, result: dict) -> ReasonerOutput:
        """Parse the agent's response into structured output."""
        raw_output = result.get("output", "")

        # For now, return a basic structure
        # In production, you'd parse the LLM output more carefully
        output = ReasonerOutput(
            claim=claim,
            causality_classification={},
            raw_response=raw_output,
        )

        # Extract structured data from the response
        # This is a simplified extraction - in production you'd use
        # output parsers or structured prompts
        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.arguments = self._extract_arguments(raw_output)

        return output

    def _extract_reasoning_chain(self, response: str) -> list[str]:
        """Extract reasoning chain from response."""
        chain = []
        lines = response.split("\n")

        in_chain = False
        for line in lines:
            line = line.strip()
            if "catena" in line.lower() or "ragionamento" in line.lower():
                in_chain = True
                continue
            if in_chain and line:
                if line.startswith(("-", "•", "*", "1", "2", "3", "4", "5")):
                    chain.append(line.lstrip("-•* 0123456789."))
                elif "→" in line or "->" in line:
                    chain.append(line)

        return chain if chain else [response[:500]]  # Fallback to truncated response

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

    def reason_with_context(
        self,
        claim: str,
        pre_retrieved_statutes: list[dict],
        pre_retrieved_precedents: list[dict],
    ) -> ReasonerOutput:
        """
        Reason about a claim with pre-retrieved context.

        Use this when statutes/precedents have already been retrieved
        by the LegalSearchPipeline.

        Args:
            claim: The legal claim.
            pre_retrieved_statutes: Already retrieved statute articles.
            pre_retrieved_precedents: Already retrieved precedents.

        Returns:
            ReasonerOutput with arguments based on provided context.
        """
        self._log("Reasoning with pre-retrieved context...")

        # Format context
        context = self._format_context(pre_retrieved_statutes, pre_retrieved_precedents)

        # Build a simpler prompt that doesn't need tools
        messages = [
            SystemMessage(content=REASONER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Analyze the following legal claim using the provided context.

CLAIM:
"{claim}"

NORMATIVE CONTEXT:
{context}

INSTRUCTIONS:
1. Classify the type of causality (Material, Legal, Concurrent Causes)
2. Build structured arguments to support the claim
3. For each argument specify: Premise, Norm, Causal Link, Conclusion
4. Provide a final reasoning chain

Respond in Italian with precise normative references."""
            ),
        ]

        # Direct LLM call without tools
        response = self.llm.invoke(messages)
        raw_output = str(response.content) if response.content else ""

        # Parse response
        output = ReasonerOutput(
            claim=claim,
            causality_classification=self._classify_from_context(claim, context),
            relevant_statutes=pre_retrieved_statutes,
            relevant_precedents=pre_retrieved_precedents,
            raw_response=raw_output,
        )

        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.arguments = self._extract_arguments(raw_output)

        self._log(
            f"Generated {len(output.arguments)} arguments from context", "success"
        )
        return output

    def _format_context(
        self,
        statutes: list[dict],
        precedents: list[dict],
    ) -> str:
        """Format retrieved context for the prompt."""
        parts = []

        if statutes:
            parts.append("LAW ARTICLES:")
            for s in statutes:
                source = "c.c." if s.get("source") == "codice_civile" else "c.p."
                parts.append(f"- Art. {s.get('articolo')} {source}: {s.get('titolo')}")
                testo = s.get("testo")
                if testo:
                    parts.append(f"  {testo[:300]}...")
            parts.append("")

        if precedents:
            parts.append("JURISPRUDENTIAL PRECEDENTS:")
            for p in precedents:
                parts.append(f"- {p.get('title', 'Untitled')}")
                summary = p.get("summary")
                if summary:
                    parts.append(f"  {summary[:200]}...")
            parts.append("")

        return "\n".join(parts)

    def _classify_from_context(self, claim: str, context: str) -> dict:
        """Classify causality type from claim and context."""
        # Use the taxonomy tool directly
        from .tools.taxonomy_tools import classify_causality_tool

        result = classify_causality_tool.invoke(
            {
                "claim": claim,
                "context": context[:1000],  # Limit context size
            }
        )

        return result if isinstance(result, dict) else {}
