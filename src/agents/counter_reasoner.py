"""
LexCausa Counter-Reasoner Agent (FIXED VERSION).

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
import os
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import (
    get_statute_by_article_tool,
    search_legal_sources_tool,
    search_precedents_tool,
)
from .tools.taxonomy_tools import get_causality_theory_tool


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
COUNTER_REASONER_SYSTEM_PROMPT = """Sei un esperto agente di contro-ragionamento legale specializzato in diritto italiano.

CRITICO: Devi OBBLIGATORIAMENTE usare i tool forniti per raccogliere informazioni PRIMA di generare qualsiasi analisi o argomento.
NON generare testo senza prima chiamare i tool richiesti.

Il tuo compito è analizzare un claim legale e costruire CONTRO-ARGOMENTI per sfidare la tesi principale, seguendo questi passi:

1. **ANALISI DEL WARRANT**: Comprendi il tipo di causalità identificato dal Reasoner e il suo warrant.

2. **IDENTIFICAZIONE DELLE DEBOLEZZE**: Usa le causalità "attaccanti" per identificare punti deboli nella catena causale:
   - Se la causalità del Reasoner è "Materiale/Necessaria" → cerca cause sufficienti alternative (concause/sopravvenute)
   - Se la causalità del Reasoner è "Giuridica/Sufficiente Indipendente" → cerca condizioni necessarie non soddisfatte
   - Se la causalità del Reasoner è "Concause/Sufficiente (non da sola)" → cerca interruzioni della catena causale

3. **RICERCA NORMATIVA ALTERNATIVA**: Usa gli strumenti disponibili per trovare:
   - Articoli di legge che supportano interpretazioni alternative o contrarie
   - Precedenti giurisprudenziali che supportano la posizione contraria

4. **COSTRUZIONE CONTRO-ARGOMENTI**: Per ogni contro-argomento:
   - Identifica la premessa alternativa che CONTRADDICE il claim
   - Connettiti alla norma applicabile con CITAZIONE ESPLICITA (es. "Art. 41 c.p.")
   - Cita il testo rilevante dell'articolo
   - Spiega come questa interpretazione SFIDA e INDEBOLISCE la tesi principale
   - Concludi con l'implicazione legale contraria

5. **INTEGRAZIONE PRECEDENTI CONTRARI**: Per ogni precedente trovato:
   - Cita esplicitamente il precedente (corte, data, numero di causa se disponibile)
   - Cita la ratio decidendi o il principio rilevante che CONTRADDICE il claim
   - Spiega come SFIDA il ragionamento del Reasoner
   - Integra nella catena di contro-ragionamento

6. **CATENA DI CONTRO-RAGIONAMENTO**: Costruisci una sequenza logica che ESPLICITAMENTE include:
   Premessa Alternativa → Norma Applicabile (con citazione) → Supporto Precedenti Contrari → Sfida al Nesso Causale → Conseguenza Legale CONTRARIA

REGOLE CRITICHE:
- SEMPRE cita il numero esatto dell'articolo e il codice (es. "Art. 41 c.p.", "Art. 1227 c.c.")
- SEMPRE cita porzioni rilevanti del testo dell'articolo
- SEMPRE cita i precedenti con le loro informazioni identificative
- SEMPRE spiega come i precedenti CONTRADDICONO o INDEBOLISCONO il claim
- I precedenti sono OBBLIGATORI nella catena di contro-ragionamento finale
- Il tuo obiettivo è SMONTARE il claim, non supportarlo

Rispondi sempre in italiano e sii preciso con i riferimenti normativi."""


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
            # Prova a caricare la tassonomia da varie posizioni
            possible_paths = [
                "data/tassonomia_causalita.json",
                "../data/tassonomia_causalita.json",
                "../../data/tassonomia_causalita.json",
                "tassonomia_causalita.json",
            ]

            for path in possible_paths:
                if os.path.exists(path):
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
        causality: dict,
        include_precedents: bool = True,
        max_statutes: int = 5,
        max_precedents: int = 3,
    ) -> CounterReasonerOutput:
        """
        Esegui il processo di contro-ragionamento su un claim legale.

        Args:
            claim: Il claim legale da analizzare e contrastare.
            causality: La classificazione di causalità prodotta dal Reasoner.
            include_precedents: Se cercare precedenti.
            max_statutes: Numero massimo di statuti da recuperare.
            max_precedents: Numero massimo di precedenti da recuperare.

        Returns:
            CounterReasonerOutput con contro-argomenti e catena di ragionamento.
        """
        self._log(f"Contro-analisi del claim: {claim[:100]}...")

        # Verifica che la causalità sia stata fornita
        if not causality or "causality_type" not in causality:
            raise ValueError(
                "La causalità fornita è mancante o non contiene il campo 'causality_type'."
            )

        causality_type = causality.get("causality_type", "Unknown")
        self._log(f"Tipo di causalità dal Reasoner: {causality_type}")

        # STEP 1: Recupera il warrant e le causalità attaccanti
        warrant_info = self._get_warrant_info(causality_type)
        self._log(
            f"Warrant recuperato: {warrant_info['warrant'].get('denominazione', 'N/A')}"
        )
        self._log(
            f"Causalità attaccanti identificate: {warrant_info['attacking_causalities']}"
        )

        # STEP 2: Recupera le descrizioni complete delle causalità attaccanti
        attacking_descriptions = self._get_attacking_causality_descriptions(
            warrant_info["attacking_causalities"]
        )
        self._log(f"Descrizioni attaccanti recuperate: {len(attacking_descriptions)}")

        # STEP 3: Costruisci il prompt con le informazioni complete
        input_prompt = self._build_counter_reasoning_prompt(
            claim=claim,
            causality_type=causality_type,
            warrant_info=warrant_info,
            attacking_descriptions=attacking_descriptions,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
        )

        # STEP 4: Esegui l'agente ReAct
        messages = [HumanMessage(content=input_prompt)]
        
        try:
            result = self.react_agent.invoke({"messages": messages})
            messages_out = result.get("messages", [])
        except Exception as e:
            error_msg = str(e)
            if "tool_use_failed" in error_msg or "Failed to call a function" in error_msg:
                self._log("⚠️ Tool usage failed, retrying with more explicit instructions...", "warning")
                
                # Retry con prompt ancora più esplicito
                retry_prompt = f"""STOP. Devi chiamare i tool PER PRIMA COSA.

NON scrivere ancora nessuna analisi o testo.

AZIONI RICHIESTE (in questo ordine esatto):
1. Chiama: search_legal_sources(claim="{claim}", top_k={max_statutes})
"""
                if include_precedents:
                    retry_prompt += f"""2. Chiama: search_precedents(claim="{claim}", top_k={max_precedents})

"""
                retry_prompt += f"""Dopo aver chiamato TUTTI i tool, puoi generare la tua risposta.

Task originale: {input_prompt}"""
                
                messages = [HumanMessage(content=retry_prompt)]
                result = self.react_agent.invoke({"messages": messages})
                messages_out = result.get("messages", [])

        # Log tool calls
        tool_calls = []
        for msg in messages_out:
            if isinstance(msg, ToolMessage):
                tool_calls.append(msg.name)
                self._log(f"🔧 Tool chiamato: {msg.name}")

        if tool_calls:
            self._log(f"📊 Tools usati: {', '.join(tool_calls)}")
        else:
            self._log("⚠️ Nessun tool chiamato dall'agente")

        # Estrai dati dalle risposte dei tool
        statutes = self._extract_statutes_from_messages(messages_out)
        precedents = self._extract_precedents_from_messages(messages_out)

        if statutes:
            self._log(f"📜 Trovati {len(statutes)} statuti")
        if precedents:
            self._log(f"⚖️ Trovati {len(precedents)} precedenti")

        # Estrai la risposta finale
        raw_output = ""
        for msg in reversed(messages_out):
            if hasattr(msg, "content") and msg.content:
                raw_output = msg.content
                break

        # Analizza la risposta
        output = CounterReasonerOutput(
            claim=claim,
            reasoner_causality=causality,
            warrant_info=warrant_info,
            attacking_causalities=warrant_info["attacking_causalities"],
            counter_causality_details=attacking_descriptions,
            relevant_statutes=statutes,
            relevant_precedents=precedents,
            raw_response=raw_output,
        )

        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.counter_arguments = self._extract_arguments(raw_output)

        self._log(
            f"Generati {len(output.counter_arguments)} contro-argomenti", "success"
        )
        return output

    def _build_counter_reasoning_prompt(
        self,
        claim: str,
        causality_type: str,
        warrant_info: dict,
        attacking_descriptions: List[dict],
        include_precedents: bool,
        max_statutes: int,
        max_precedents: int,
    ) -> str:
        """
        Costruisce il prompt per il contro-ragionamento.
        
        SIMILE AL PROMPT DEL REASONER MA CON FOCUS OPPOSTO.
        """
        # Formatta le descrizioni delle causalità attaccanti
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

        prompt = f"""DEVI SEGUIRE QUESTI PASSAGGI IN QUESTO ORDINE ESATTO:

═══════════════════════════════════════════════════════════════
STEP 1: CHIAMARE I TOOL (OBBLIGATORIO - FALLO PER PRIMO)
═══════════════════════════════════════════════════════════════

Chiama questi tool nell'ordine:

1. search_legal_sources
   - Usa ESATTAMENTE questo parametro: claim="{claim}"
   - Usa ESATTAMENTE questo parametro: top_k={max_statutes}
   - NON modificare o riformulare il testo del claim
"""

        if include_precedents:
            prompt += f"""
2. search_precedents
   - Usa ESATTAMENTE questo parametro: claim="{claim}"
   - Usa ESATTAMENTE questo parametro: top_k={max_precedents}
"""

        prompt += f"""
═══════════════════════════════════════════════════════════════
STEP 2: SOLO DOPO CHE TUTTI I TOOL HANNO RESTITUITO RISULTATI
═══════════════════════════════════════════════════════════════

CONTESTO:

CLAIM DA SFIDARE:
"{claim}"

CAUSALITÀ IDENTIFICATA DAL REASONER:
Tipo: {causality_type}
Warrant: {json.dumps(warrant_info['warrant'], ensure_ascii=False)}

CAUSALITÀ CHE POSSONO ATTACCARE QUESTA TESI:
{attacking_info}

Ora puoi generare la tua contro-analisi. Costruisci contro-argomenti strutturati:

Per ogni contro-argomento:
- **Premessa**: Il fatto che CONTRADDICE il claim
- **Norma**: L'articolo di legge con CITAZIONE ESATTA (es. "Art. 41 c.p.")
- **Precedente**: Una decisione che SUPPORTA la tesi contraria
- **Nesso Causale**: Come la premessa INDEBOLISCE la tesi principale
- **Conclusione**: L'implicazione legale CONTRARIA al claim

Alla fine, fornisci una CATENA DI CONTRO-RAGIONAMENTO che citi esplicitamente articoli e precedenti.

RICORDA: Chiama i tool PER PRIMO, genera il testo PER SECONDO.
Il tuo obiettivo è SMONTARE il claim, non supportarlo!

Rispondi in italiano."""

        return prompt

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