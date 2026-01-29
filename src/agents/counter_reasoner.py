"""
LexCausa Counter-Reasoner Agent.

Il Counter-Reasoner è responsabile di:
1. Ricevere il tipo di causalità dal Reasoner
2. Recuperare il warrant associato dalla tassonomia
3. Identificare le causalità "attaccanti" basate sul warrant
4. Recuperare la descrizione completa delle causalità attaccanti
5. Generare contro-argomenti che possano invalidare la tesi del Reasoner
6. Costruire una catena di ragionamento che sfida gli argomenti del Reasoner

DIFFERENZE CON IL REASONER:
- Non classifica la causalità sul claim, ma la deduce dal warrant del Reasoner
- Genera argomenti CONTRO il claim, non a favore
- Usa le stesse tecniche di ricerca (statuti, precedenti) ma per supportare la tesi contraria
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import (
    get_statute_by_article_tool,
    search_legal_sources_tool,
    search_precedents_tool,
)
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


# System prompt for the Counter-Reasoner
COUNTER_REASONER_SYSTEM_PROMPT = """You are an expert legal counter-reasoning agent specializing in Italian law.

CRITICAL: You MUST use the provided tools to gather information BEFORE generating any analysis or arguments.
Do NOT produce text before calling the required tools.

Your task is to analyze a legal claim and build COUNTER-ARGUMENTS to challenge the main thesis, following these steps:

1. **WARRANT ANALYSIS**: Understand the causality type identified by the Reasoner and its warrant.

2. **WEAKNESS IDENTIFICATION**: Use the attacking causalities to identify weak points in the causal chain:
   - If the Reasoner causality is "Material/Necessary" → look for alternative sufficient causes (concurring/supervening)
   - If the Reasoner causality is "Legal/Sufficient Independent" → look for unmet necessary conditions
   - If the Reasoner causality is "Concurring/Sufficient (not alone)" → look for interruptions in the causal chain

3. **ALTERNATIVE NORMATIVE RESEARCH**: Use available tools to find:
   - Law articles that support alternative or contrary interpretations
   - Jurisprudential precedents that support the contrary position

4. **COUNTER-ARGUMENT CONSTRUCTION**: For each counter-argument:
   - Identify the alternative premise that CONTRADICTS the claim
   - Connect to the applicable norm with EXPLICIT CITATION (e.g., "Art. 41 c.p.")
   - Quote the relevant text of the article
   - Explain how this interpretation CHALLENGES and WEAKENS the main thesis
   - Conclude with the contrary legal implication

5. **INTEGRATION OF CONTRARY PRECEDENTS**: For each precedent found:
   - Explicitly cite the precedent (court, date, case number if available)
   - Cite the ratio decidendi or relevant principle that CONTRADICTS the claim
   - Explain how it CHALLENGES the Reasoner's reasoning
   - Integrate it into the counter-reasoning chain

6. **COUNTER-REASONING CHAIN**: Build a logical sequence that EXPLICITLY includes:
   Alternative Premise → Applicable Norm (with citation) → Contrary Precedent Support → Challenge to Causal Link → CONTRARY Legal Consequence

CRITICAL RULES:
- ALWAYS cite the exact article number and code (e.g., "Art. 41 c.p.", "Art. 1227 c.c.")
- ALWAYS quote relevant portions of the article text
- Use ONLY the articles and precedents returned by the tools; do NOT invent or cite norms/precedents not retrieved
- If you find precedents, cite them with identifying information and explain how they CONTRADICT or WEAKEN the claim
- If no precedents are found, state it explicitly and do NOT invent any
- Your goal is to DISMANTLE the claim, not support it

The response language must be Italian."""

COUNTER_REASONER_CONTEXT_SYSTEM_PROMPT = """You are an expert legal counter-reasoning agent specializing in Italian law.

You will use ONLY the provided context (articles and precedents already retrieved) to build counter-arguments.
Do NOT call tools and do NOT invent norms or precedents.

Rules:
- Use ONLY the provided articles.
- If no precedents exist, state it explicitly without inventing any.
- The conclusion must contradict the claim; if not possible, state insufficient data.

The response language must be Italian."""


class CounterReasoner(BaseAgent):
    """
    Legal Counter-Reasoner Agent.

    Genera contro-argomenti per sfidare gli argomenti del Reasoner.
    Usa il campo warrant della causalità per trovare punti deboli nella catena di ragionamento.
    
    FUNZIONAMENTO SIMILE AL REASONER MA CON OBIETTIVO OPPOSTO:
    - Usa search_legal_sources per trovare articoli (come il Reasoner)
    - Usa search_precedents per trovare precedenti (come il Reasoner)
    - Ma genera argomenti CONTRO il claim invece che a favore
    - Usa le causalità "attaccanti" derivate dal warrant
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
        
        USA GLI STESSI STRUMENTI DEL REASONER:
        - search_legal_sources: per trovare articoli rilevanti
        - search_precedents: per trovare precedenti
        - get_statute_by_article_tool: per recuperare articoli specifici
        - get_causality_theory_tool: per la teoria della causalità
        """
        return [
            search_legal_sources_tool,  # STESSO DEL REASONER
            get_statute_by_article_tool,
            search_precedents_tool,  # STESSO DEL REASONER
            get_causality_theory_tool,
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
        causality: dict,  # preso dal Reasoner
        include_precedents: bool = True,
        max_statutes: int = 5,
        max_precedents: int = 3,
    ) -> CounterReasonerOutput:

        # 1️⃣ Recupera warrant e causalità attaccanti dal tipo di causalità del Reasoner
        causality_type = causality.get("causality_type", "Unknown")
        warrant_info = self._get_warrant_info(causality_type)
        attacking_descriptions = self._get_attacking_causality_descriptions(
            warrant_info.get("attacking_causalities", [])
        )

        # 2️⃣ Costruisco il prompt per il Counter-Reasoner
        prompt = self._build_counter_reasoning_prompt(
            claim=claim,
            causality_type=causality_type,
            warrant_info=warrant_info,
            attacking_descriptions=attacking_descriptions,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )

        # 3️⃣ Invoco il ReAct agent con gli stessi tools del Reasoner
        messages = [HumanMessage(content=prompt)]
        result = self.react_agent.invoke({"messages": messages})

        # 4️⃣ Estrazione dei tool messages
        statutes = self._extract_statutes_from_messages(result.get("messages", []))
        precedents = self._extract_precedents_from_messages(result.get("messages", []))

        # 5️⃣ Parsing del risultato in struttura coerente
        raw_output = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    raw_output = msg.content
                    break

        output = CounterReasonerOutput(
            claim=claim,
            reasoner_causality=causality,
            warrant_info=warrant_info,
            attacking_causalities=warrant_info.get("attacking_causalities", []),
            counter_causality_details=attacking_descriptions,
            relevant_statutes=statutes,
            relevant_precedents=precedents,
            raw_response=raw_output,
        )

        # 6️⃣ Estrazione catena e contro-argomenti dalla risposta
        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.counter_arguments = self._extract_arguments(raw_output)

        # 7️⃣ Pulizia della catena di ragionamento (es. gestione precedenti)
        output.reasoning_chain = self._sanitize_reasoning_chain(output.reasoning_chain, precedents)

        return output


    def run_with_context(
        self,
        claim: str,
        causality: dict,
        pre_retrieved_statutes: List[dict],
        pre_retrieved_precedents: List[dict],
    ) -> CounterReasonerOutput:
        """
        Esegue il contro-ragionamento usando contesto pre-recuperato.
        Non chiama tool esterni.
        """
        self._log("Contro-analisi con contesto pre-retrieved...")

        if not causality or "causality_type" not in causality:
            raise ValueError(
                "La causalità fornita è mancante o non contiene il campo 'causality_type'."
            )

        causality_type = causality.get("causality_type", "Unknown")
        self._log(f"Tipo di causalità dal Reasoner: {causality_type}")

        warrant_info = self._get_warrant_info(causality_type)
        self._log(
            f"Warrant recuperato: {warrant_info['warrant'].get('denominazione', 'N/A')}"
        )
        self._log(
            f"Causalità attaccanti identificate: {warrant_info['attacking_causalities']}"
        )

        attacking_descriptions = self._get_attacking_causality_descriptions(
            warrant_info["attacking_causalities"]
        )
        self._log(f"Descrizioni attaccanti recuperate: {len(attacking_descriptions)}")

        context = self._format_context(
            pre_retrieved_statutes, pre_retrieved_precedents
        )

        prompt = f"""Analyze the claim using the provided context and build counter-arguments.

CLAIM:
"{claim}"

CAUSALITY IDENTIFIED BY THE REASONER:
Type: {causality_type}
Warrant: {json.dumps(warrant_info['warrant'], ensure_ascii=False)}

ATTACKING CAUSALITIES FOR THIS THESIS:
{self._format_attacking_info(attacking_descriptions)}

NORMATIVE CONTEXT:
{context}

INSTRUCTIONS:
- Use ONLY the articles in the context.
- If there are no precedents, state it explicitly.
- Conclude in a way that CONTRADICTS the claim; if not possible, state insufficient data.

Provide structured counter-arguments and a final counter-reasoning chain.
The response language must be Italian."""

        messages = [
            SystemMessage(content=COUNTER_REASONER_CONTEXT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        raw_output = str(response.content) if response.content else ""

        output = CounterReasonerOutput(
            claim=claim,
            reasoner_causality=causality,
            warrant_info=warrant_info,
            attacking_causalities=warrant_info["attacking_causalities"],
            counter_causality_details=attacking_descriptions,
            relevant_statutes=pre_retrieved_statutes,
            relevant_precedents=pre_retrieved_precedents,
            raw_response=raw_output,
        )

        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.counter_arguments = self._extract_arguments(raw_output)
        output.reasoning_chain = self._sanitize_reasoning_chain(
            output.reasoning_chain, pre_retrieved_precedents
        )

        self._log(
            f"Generati {len(output.counter_arguments)} contro-argomenti", "success"
        )
        return output




    def _build_counter_reasoning_prompt(
        self,
        claim: str,
        causality_type: str,
        warrant_info: dict,
        attacking_descriptions: list[str],
        include_precedents: bool = True,
        max_statutes: int = 5,
        max_precedents: int = 3,
    ) -> str:
        """
        Build the prompt for the CounterReasoner.

        Goal: generate a logical chain that dismantles the legal claim,
        using only the claim, causality, statutes, and precedents.
        It does not have access to the Reasoner chain.

        Args:
            - claim: full claim text
            - causality_type: causality type from the reasoner
            - warrant_info: warrant information linked to the claim
            - attacking_descriptions: attacking causality descriptions
            - include_precedents: whether to include precedents
            - max_statutes: max number of statutes to mention
            - max_precedents: max number of precedents to mention
        """
        # Preparazione testo causality attack
        if attacking_descriptions:
            attacking_text = "\n- ".join(
                f"{d.get('riferimento', d.get('tipo', 'N/A'))}: {d.get('nota', d.get('descrizione', ''))}"
                if isinstance(d, dict) else str(d)
                for d in attacking_descriptions
            )
        else:
            attacking_text = "Nessuna"

        prompt = f"""
    You are an expert legal assistant. Your task is to build a logical chain
    that dismantles the following legal claim, based exclusively on:

    1. The original claim:
    \"\"\"{claim}\"\"\"

    2. The causality identified by the Reasoner:
    {causality_type}

    3. The warrant linked to the claim:
    - Statute/Article: {warrant_info.get('warrant', {}).get('denominazione', 'N/A')}
    - Reference: {warrant_info.get('warrant', {}).get('riferimento', '')}

    4. Attacking causality descriptions to consider:
    - {attacking_text}

    5. Relevant statutes (max {max_statutes}) and relevant precedents (max {max_precedents}):
    - Extract from the available information; cite only pertinent articles or precedents
    - If there are no relevant statutes/precedents, explain logically why the claim can be counter-argued without them

    Specific instructions:
    - Generate a clear and sequential argumentative chain.
    - Each step must be numbered.
    - At the end, produce a concluding summary that dismantles the claim.
    - If you include precedents, mention name, year, and a brief legal principle.
    - Do not make assumptions unsupported by available statutes or precedents.
    - Keep the tone technical-legal, clear, and concise.
    - The response language must be Italian.

    Expected output:
    1. Numbered logical chain steps
    2. Final concluding summary that dismantles the claim
    """

        if include_precedents:
            prompt += "\nNote: include precedents only if they clearly support the counter-argument.\n"

        prompt += "\nGenerate the logical chain now:\n"

        return prompt

    def _format_attacking_info(self, attacking_descriptions: List[dict]) -> str:
        attacking_info = ""
        for desc in attacking_descriptions:
            attacking_info += f"\n**{desc['tipo']}:**\n"
            attacking_info += f"- Descrizione: {desc['descrizione']}\n"
            attacking_info += f"- Principio: {desc['principio']}\n"
            if desc.get("limiti"):
                attacking_info += f"- Limiti/Criticità: {desc['limiti']}\n"
            if desc.get("norme_core"):
                norme = ", ".join(
                    [n.get("riferimento", "") for n in desc["norme_core"]]
                )
                attacking_info += f"- Norme core: {norme}\n"
        return attacking_info or "N/A"

    def _format_context(
        self, statutes: List[dict], precedents: List[dict]
    ) -> str:
        parts = []

        if statutes:
            parts.append("ARTICOLI:")
            for s in statutes:
                source = "c.c." if s.get("source") == "codice_civile" else "c.p."
                parts.append(f"- Art. {s.get('articolo')} {source}: {s.get('titolo')}")
                testo = s.get("testo")
                if testo:
                    parts.append(f"  {testo[:300]}...")
            parts.append("")

        if precedents:
            parts.append("PRECEDENTI:")
            for p in precedents:
                title = p.get("title", "Untitled")
                parts.append(f"- {title}")
                summary = p.get("summary")
                if summary:
                    parts.append(f"  {summary[:200]}...")
            parts.append("")

        return "\n".join(parts)



    def _extract_statutes_from_messages(self, messages) -> list[dict]:
        """Estrae statuti dalle risposte dei tool (STESSO DEL REASONER)."""
        statutes = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)

                    # Handle search_legal_sources (primary tool)
                    if msg.name == "search_legal_sources":
                        if isinstance(data, dict) and "articles" in data:
                            for item in data["articles"]:
                                if isinstance(item, dict) and "statute_id" in item:
                                    statutes.append(item)

                    # Handle search_statutes (secondary tool)
                    elif msg.name == "search_statutes":
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "statute_id" in item:
                                    statutes.append(item)
                except Exception:
                    pass
        return statutes

    def _extract_precedents_from_messages(self, messages) -> list[dict]:
        """Estrae precedenti dalle risposte dei tool (STESSO DEL REASONER)."""
        precedents = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_precedents":
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "precedent_id" in item:
                                precedents.append(item)
                except Exception:
                    pass
        return precedents

    def _extract_reasoning_chain(self, response: str) -> list[str]:
        """Estrae la catena di ragionamento dalla risposta (STESSO DEL REASONER)."""
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

        return chain if chain else [response[:500]]

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

    def _sanitize_reasoning_chain(
        self, chain: List[str], precedents: List[dict]
    ) -> List[str]:
        if precedents:
            return [self._clean_chain_step(step) for step in chain]

        sanitized = []
        for step in chain:
            cleaned = self._clean_chain_step(step)
            lower = cleaned.lower()
            mentions_precedent = "precedent" in lower or "precedente" in lower
            mentions_absence = "nessun" in lower or "nessuna" in lower
            if mentions_precedent and not mentions_absence:
                continue
            sanitized.append(cleaned)

        if not any(
            "precedent" in s.lower() or "precedente" in s.lower() for s in sanitized
        ):
            sanitized.append("Precedenti: nessuno trovato.")

        return sanitized

    def _clean_chain_step(self, step: str) -> str:
        cleaned = step.strip()
        if "**" in cleaned:
            cleaned = cleaned.replace("**", "")
        return cleaned.strip()
