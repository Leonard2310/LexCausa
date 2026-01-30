"""
LexCausa Counter-Reasoner Agent.

Il Counter-Reasoner è responsabile di:
1. Ricevere il tipo di causalità dal Reasoner
2. Recuperare il warrant associato dalla tassonomia
3. Identificare le causalità "attaccanti" basate sul warrant
4. Recuperare la descrizione completa delle causalità attaccanti
5. Generare contro-argomenti che possano invalidare la tesi del Reasoner
6. Costruire una catena di ragionamento che sfida gli argomenti del Reasoner

IMPORTANT: The CounterReasoner does NOT search for articles/precedents itself.
The pre-retrieval is done by api_server using LegalSearchPipeline.
This ensures the agent bases its reasoning ONLY on the retrieved knowledge.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import get_statute_by_article_tool
from .tools.taxonomy_tools import get_causality_theory_tool
from config import settings


@dataclass
class CounterArgument:
    """Structured representation of a counter-argument."""

    premise: str = ""
    norm: str = ""
    link: str = ""
    conclusion: str = ""


@dataclass
class CounterReasonerOutput:
    """Structured output from the Counter-Reasoner."""

    claim: str
    reasoner_causality: dict
    warrant_info: dict = field(default_factory=dict)
    attacking_causalities: List[str] = field(default_factory=list)
    counter_causality_details: List[dict] = field(default_factory=list)
    relevant_statutes: List[dict] = field(default_factory=list)
    relevant_precedents: List[dict] = field(default_factory=list)
    counter_arguments: List[CounterArgument] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "reasoner_causality": self.reasoner_causality,
            "warrant_info": self.warrant_info,
            "attacking_causalities": self.attacking_causalities,
            "counter_causality_details": self.counter_causality_details,
            "statutes": self.relevant_statutes,
            "precedents": self.relevant_precedents,
            "counter_arguments": [
                {
                    "premise": arg.premise,
                    "norm": arg.norm,
                    "link": arg.link,
                    "conclusion": arg.conclusion,
                }
                for arg in self.counter_arguments
            ],
            "reasoning_chain": self.reasoning_chain,
            "raw_response": self.raw_response,
        }


# System prompt for the Counter-Reasoner (with pre-retrieved context)
COUNTER_REASONER_SYSTEM_PROMPT = """You are an expert legal counter-reasoning agent specializing in Italian law.

You will receive a legal claim along with PRE-RETRIEVED articles and precedents as your KNOWLEDGE BASE.
Your task is to build COUNTER-ARGUMENTS to challenge the main thesis using ONLY the provided knowledge.

CRITICAL RULES:
- Use ONLY the articles provided in the KNOWLEDGE BASE - do NOT invent or cite articles not provided
- Use ONLY the precedents provided - do NOT invent precedents
- If no precedents are provided, explicitly state this and proceed without them
- Your goal is to DISMANTLE the claim, not support it
- Even if the causality theory lists core/accessory norms, DO NOT cite them unless the article appears in the KNOWLEDGE BASE above
- ALWAYS cite exact article numbers and codes (e.g., "Art. 41 c.p.", "Art. 1227 c.c.")

Your task follows these steps:

1. **WARRANT ANALYSIS**: Understand the causality type from the Reasoner and its warrant.

2. **GET CAUSAL THEORY**: Use `get_causality_theory` to retrieve theory for attacking causalities.

3. **WEAKNESS IDENTIFICATION**: Identify weak points in the causal chain using the attacking causalities.

4. **COUNTER-ARGUMENT CONSTRUCTION**: For each counter-argument:
   - **Premessa Alternativa**: An alternative premise that CONTRADICTS the claim
   - **Norma**: Article citation with quoted text FROM THE KNOWLEDGE BASE
   - **Nesso Causale Alternativo**: How this challenges the original causal link
   - **Conclusione Contraria**: The CONTRARY legal implication

5. **COUNTER-REASONING CHAIN**: Build a logical sequence:
   Alternative Premise → Applicable Norm → Challenge to Causal Link → CONTRARY Legal Consequence

The response language must be Italian."""


class CounterReasoner(BaseAgent):
    """
    Legal Counter-Reasoner Agent.

    Genera contro-argomenti per sfidare gli argomenti del Reasoner.
    Usa il campo warrant della causalità per trovare punti deboli nella catena di ragionamento.
    
    Flow:
    1. api_server pre-retrieves statutes and precedents
    2. CounterReasoner.run() receives the causality from Reasoner + pre-retrieved knowledge
    3. ReAct agent uses tools (get_causality_theory) to build counter-arguments
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Counter-Reasoner agent."""
        super().__init__(config)
        self._react_agent = None
        self._taxonomy = None

    def _load_taxonomy(self) -> dict:
        """Carica la tassonomia di causalità."""
        if self._taxonomy is None:
            candidate_paths = [
                settings.taxonomy_path,
                settings.data_dir / "tassonomia_causalita.json",
                Path("tassonomia_causale.json"),
                Path("tassonomia_causalita.json"),
            ]

            for path in candidate_paths:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        self._taxonomy = json.load(f)
                    self._log(f"Tassonomia caricata da: {path}", "success")
                    break

            if self._taxonomy is None:
                self._log(
                    "⚠️ Tassonomia non trovata, uso struttura vuota", "warning"
                )
                self._taxonomy = {"tassonomia_causalita": []}

        return self._taxonomy

    def _get_warrant_info(self, causality_type: str) -> dict:
        """
        Estrae le informazioni del warrant dalla tassonomia.

        Args:
            causality_type: Il tipo di causalità (es. "Materiale", "Giuridica", "Concause / Sopravvenute")

        Returns:
            Dict contenente:
            - warrant: dict con denominazione e todo_nli
            - attacking_causalities: lista dei tipi di causalità che possono attaccare questa
            - full_details: dettagli completi della causalità dalla tassonomia
        """
        taxonomy = self._load_taxonomy()

        for entry in taxonomy.get("tassonomia_causalita", []):
            if entry.get("tipo_causalita") == causality_type:
                warrant = entry.get("warrant", {})
                warrant_denominazione = warrant.get("denominazione", "")

                # Determina le causalità attaccanti basate sul warrant
                attacking_causalities = []

                if "Necessaria" in warrant_denominazione:
                    # La causalità necessaria può essere attaccata da cause sufficienti alternative
                    attacking_causalities.append("Concause / Sopravvenute")
                elif "Sufficiente Indipendente" in warrant_denominazione:
                    # La causalità sufficiente indipendente può essere attaccata da condizioni necessarie
                    attacking_causalities.append("Materiale")
                elif "Sufficiente (non da sola)" in warrant_denominazione:
                    # Le concause possono essere attaccate da entrambe
                    attacking_causalities.extend(["Materiale", "Giuridica"])

                return {
                    "warrant": warrant,
                    "attacking_causalities": attacking_causalities,
                    "full_details": entry,
                }

        return {
            "warrant": {},
            "attacking_causalities": [],
            "full_details": {},
        }

    def _get_attacking_causality_descriptions(
        self, attacking_types: List[str]
    ) -> List[dict]:
        """
        Recupera le descrizioni complete delle causalità attaccanti.

        Args:
            attacking_types: Lista dei tipi di causalità attaccanti

        Returns:
            Lista di dict con dettagli completi di ogni causalità attaccante
        """
        taxonomy = self._load_taxonomy()
        descriptions = []

        for entry in taxonomy.get("tassonomia_causalita", []):
            if entry.get("tipo_causalita") in attacking_types:
                descriptions.append(
                    {
                        "tipo": entry.get("tipo_causalita"),
                        "descrizione": entry.get("descrizione_ruolo", ""),
                        "principio": entry.get("principio_test_applicato", ""),
                        "limiti": entry.get("limiti_criticita", ""),
                        "norme_core": entry.get("norme_core", []),
                        "norme_accessorie": entry.get("norme_accessorie", []),
                        "warrant": entry.get("warrant", {}),
                    }
                )

        return descriptions

    @property
    def tools(self) -> list:
        """
        Get the tools available to this agent.
        
        NOTE: No search tools - the agent works with pre-retrieved context.
        Only taxonomy tools for causality theory retrieval.
        """
        return [
            get_causality_theory_tool,
            get_statute_by_article_tool,  # For looking up specific articles by number
        ]

    @property
    def react_agent(self):
        """Lazy initialization of the ReAct agent using LangGraph."""
        if self._react_agent is None:
            self._react_agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=COUNTER_REASONER_SYSTEM_PROMPT,
            )
        return self._react_agent

    def run(
        self,
        claim: str,
        causality: dict,
        pre_retrieved_statutes: List[dict],
        pre_retrieved_precedents: List[dict],
    ) -> CounterReasonerOutput:
        """
        Execute the counter-reasoning process with pre-retrieved knowledge.

        Args:
            claim: The legal claim to counter-argue.
            causality: Causality classification from the Reasoner.
            pre_retrieved_statutes: Already retrieved and filtered statute articles.
            pre_retrieved_precedents: Already retrieved precedents.

        Returns:
            CounterReasonerOutput with counter-arguments and reasoning chain.
        """
        self._log(f"Counter-analyzing claim: {claim[:100]}...")
        self._log(f"📚 Knowledge base: {len(pre_retrieved_statutes)} statutes, {len(pre_retrieved_precedents)} precedents")

        if not causality or "causality_type" not in causality:
            self._log("⚠️ Causality not provided or invalid", "warning")
            causality = {"causality_type": "Unknown"}

        causality_type = causality.get("causality_type", "Unknown")
        self._log(f"🎯 Causality from Reasoner: {causality_type}")

        # Get warrant and attacking causalities
        warrant_info = self._get_warrant_info(causality_type)
        self._log(f"🛡️ Warrant: {warrant_info['warrant'].get('denominazione', 'N/A')}")
        self._log(f"⚔️ Attacking causalities: {warrant_info['attacking_causalities']}")

        attacking_descriptions = self._get_attacking_causality_descriptions(
            warrant_info["attacking_causalities"]
        )

        # Enrich with relevant taxonomy norms for attacking causalities
        taxonomy_statutes: list[dict] = []

        for att_type in warrant_info.get("attacking_causalities", []):
            theory = get_causality_theory_tool.invoke(
                {"causality_type": att_type, "claim": claim}
            )
            core_rel = theory.get("norme_core_rilevanti", [])
            acc_rel = theory.get("norme_accessorie_rilevanti", [])
            core_full = theory.get("norme_core", [])
            acc_full = theory.get("norme_accessorie", [])
            norms = core_rel + acc_rel

            kept_refs = [n.get("riferimento") for n in norms if n.get("riferimento")]
            kept_set = {r for r in kept_refs if r}
            discarded_refs = [
                n.get("riferimento")
                for n in (core_full + acc_full)
                if n.get("riferimento") and n.get("riferimento") not in kept_set
            ]
            self._log(
                f"🔎 [taxonomy] Causalità {att_type}: core {len(core_rel)}/{len(core_full)}, accessorie {len(acc_rel)}/{len(acc_full)}"
            )
            if kept_refs:
                self._log(f"   ✔️ Tenute: {', '.join(kept_refs)}")
            if discarded_refs:
                self._log(f"   ❌ Scartate: {', '.join(discarded_refs)}")

            taxonomy_statutes.extend([self._norm_to_statute_dict(n) for n in norms])

        all_statutes = pre_retrieved_statutes + taxonomy_statutes
        # Deduplicate
        seen_keys = set()
        deduped_statutes = []
        for s in all_statutes:
            key = (s.get("articolo"), s.get("source"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_statutes.append(s)

        # Format knowledge base for prompt
        knowledge_base = self._format_context_for_prompt(
            deduped_statutes, 
            pre_retrieved_precedents
        )

        allowed_statutes = [
            f"Art. {s.get('articolo')} ({'c.c.' if s.get('source') == 'codice_civile' else 'c.p.'})"
            for s in deduped_statutes
        ]
        allowed_precedents = [p.get("title", "Untitled") for p in pre_retrieved_precedents]

        # Build prompt with context
        input_prompt = self._build_counter_reasoning_prompt_with_context(
            claim=claim,
            causality_type=causality_type,
            warrant_info=warrant_info,
            attacking_descriptions=attacking_descriptions,
            knowledge_base=knowledge_base,
            allowed_statutes=allowed_statutes,
            allowed_precedents=allowed_precedents,
        )

        # Execute the ReAct agent
        messages = [HumanMessage(content=input_prompt)]
        try:
            result = self.react_agent.invoke({"messages": messages})
            messages_out = result.get("messages", [])
        except Exception as e:
            # Graceful fallback to avoid breaking the pipeline when tool calls fail
            error_msg = f"Errore durante l'esecuzione del Counter-Reasoner: {e}"
            self._log(error_msg, "error")
            return CounterReasonerOutput(
                claim=claim,
                reasoner_causality=causality,
                warrant_info=warrant_info,
                attacking_causalities=warrant_info["attacking_causalities"],
                counter_causality_details=attacking_descriptions,
                relevant_statutes=pre_retrieved_statutes,
                relevant_precedents=pre_retrieved_precedents,
                counter_arguments=[],
                reasoning_chain=[error_msg],
                raw_response=error_msg,
            )

        # Log tool calls
        tool_names = []
        for msg in messages_out:
            if hasattr(msg, 'name') and msg.name:
                tool_names.append(msg.name)
                self._log(f"🔧 Tool called: {msg.name}")

        if tool_names:
            self._log(f"📊 Tools used: {', '.join(set(tool_names))}")

        # Get the final response
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
        output = CounterReasonerOutput(
            claim=claim,
            reasoner_causality=causality,
            warrant_info=warrant_info,
            attacking_causalities=warrant_info["attacking_causalities"],
            counter_causality_details=attacking_descriptions,
            relevant_statutes=deduped_statutes,
            relevant_precedents=pre_retrieved_precedents,
            raw_response=raw_output,
        )

        # Parse response
        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.counter_arguments = self._extract_arguments(raw_output)
        output.reasoning_chain = self._sanitize_reasoning_chain(
            output.reasoning_chain, 
            pre_retrieved_precedents
        )

        self._log(f"✅ Generated {len(output.counter_arguments)} counter-arguments", "success")
        return output

    def _build_counter_reasoning_prompt_with_context(
        self,
        claim: str,
        causality_type: str,
        warrant_info: dict,
        attacking_descriptions: List[dict],
        knowledge_base: str,
        allowed_statutes: List[str],
        allowed_precedents: List[str],
    ) -> str:
        """
        Build the prompt for CounterReasoner with pre-retrieved context.
        """
        attacking_text = self._format_attacking_info(attacking_descriptions)
        statutes_list = "\n".join(f"- {a}" for a in allowed_statutes) or "- Nessun articolo disponibile"
        precedents_list = "\n".join(f"- {p}" for p in allowed_precedents) or "- Nessun precedente disponibile"

        return f"""Analyze the following legal claim and build COUNTER-ARGUMENTS to dismantle it.

CLAIM:
"{claim}"

CAUSALITY IDENTIFIED BY THE REASONER:
Type: {causality_type}
Warrant: {warrant_info.get('warrant', {}).get('denominazione', 'N/A')}

ATTACKING CAUSALITIES TO EXPLOIT:
{attacking_text}

=== KNOWLEDGE BASE (Your ONLY source of articles and precedents) ===
{knowledge_base}
=== END KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
{statutes_list}

ALLOWED PRECEDENT REFERENCES (do not cite others):
{precedents_list}

INSTRUCTIONS:
1. Use `get_causality_theory` to understand the attacking causalities theories.
2. Build counter-arguments that CHALLENGE and DISMANTLE the claim.
3. Each counter-argument must have:
   - Premessa Alternativa: An alternative premise that contradicts the claim
   - Norma: Article citation with quoted text FROM THE KNOWLEDGE BASE (only from ALLOWED STATUTE REFERENCES; if none apply, omit the norma field)
   - Nesso Causale Alternativo: How this challenges the causal link
   - Conclusione Contraria: The contrary legal implication
4. End with a REASONING CHAIN that shows how your counter-arguments dismantle the claim.

CRITICAL: Use ONLY the articles and precedents in the KNOWLEDGE BASE above.
Do NOT invent or cite articles not provided. If a needed article/precedent is absent from the allowed lists, omit it instead of stating it is unavailable.

Generate your counter-analysis now (in Italian):"""

    def _format_attacking_info(self, attacking_descriptions: List[dict]) -> str:
        attacking_info = ""
        for desc in attacking_descriptions:
            attacking_info += f"\n**{desc['tipo']}:**\n"
            attacking_info += f"- Descrizione: {desc['descrizione']}\n"
            attacking_info += f"- Principio: {desc['principio']}\n"
            if desc.get("limiti"):
                attacking_info += f"- Limiti/Criticità: {desc['limiti']}\n"
            # Norme core/accessorie non riportate per evitare citazioni fuori dal knowledge base
        return attacking_info or "N/A"

    def _extract_arguments(self, response: str) -> List[CounterArgument]:
        """Estrae contro-argomenti strutturati dalla risposta."""
        arguments = []
        sections = response.split("**")
        current_arg = {}

        for section in sections:
            section = section.strip()
            lower_section = section.lower()

            if "premessa" in lower_section:
                if current_arg:
                    # Crea CounterArgument con valori di default per campi mancanti
                    arguments.append(
                        CounterArgument(
                            premise=current_arg.get("premise", ""),
                            norm=current_arg.get("norm", ""),
                            link=current_arg.get("link", ""),
                            conclusion=current_arg.get("conclusion", ""),
                        )
                    )
                current_arg = {"premise": section}
            elif "norma" in lower_section:
                current_arg["norm"] = section
            elif "nesso" in lower_section:
                current_arg["link"] = section
            elif "conclusione" in lower_section:
                current_arg["conclusion"] = section

        if current_arg:
            arguments.append(
                CounterArgument(
                    premise=current_arg.get("premise", ""),
                    norm=current_arg.get("norm", ""),
                    link=current_arg.get("link", ""),
                    conclusion=current_arg.get("conclusion", ""),
                )
            )

        return arguments
