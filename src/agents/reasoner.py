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

from dataclasses import dataclass, field
from typing import Optional
import json
from langchain_core.messages import ToolMessage

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import (
    get_statute_by_article_tool,
    search_precedents_tool,
    search_statutes_tool,
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
   - Connect to the applicable norm
   - Explain the causal link
   - Conclude with the legal implication

5. **REASONING CHAIN**: Build a logical sequence that connects:
   Facts → Causal Link → Norm → Legal Consequence

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
            search_statutes_tool,
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
        causality = self._extract_causality_from_messages(messages_out)

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

INSTRUCTIONS:
1. First, classify the type of causality using the `classify_causality` tool
2. Search for relevant law articles using `search_statutes` (max {max_statutes} results)
"""

        if include_precedents:
            prompt += f"""3. Search for relevant precedents using `search_precedents` (max {max_precedents} results)
4. Retrieve the complete causal theory using `get_causality_theory`
5. Build structured arguments to support the claim
"""
        else:
            prompt += """3. Retrieve the complete causal theory using `get_causality_theory`
4. Build structured arguments to support the claim
"""

        prompt += """
For each argument, specify:
- **Premise**: The starting fact or situation
- **Norm**: The applicable law article (with precise reference)
- **Causal Link**: How the premise connects to the norm
- **Conclusion**: The legal implication

At the end, provide a REASONING CHAIN that synthesizes the logical path.

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
