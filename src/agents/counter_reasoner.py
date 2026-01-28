"""
LexCausa Counter-Reasoner Agent

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

from langchain_core.messages import HumanMessage, ToolMessage
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
- Usa SOLO gli articoli e i precedenti restituiti dai tool; NON inventare o citare norme o precedenti non recuperati
- Se trovi precedenti dai tool, citali con le loro informazioni identificative e spiega come CONTRADDICONO o INDEBOLISCONO il claim
- Se non trovi precedenti, dichiaralo esplicitamente e NON inventarne
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
        Costruisce il prompt per il CounterReasoner.

        Obiettivo: generare una catena logica che smonta il claim legale,
        usando solo il claim, causalità, statuti e precedenti. 
        Non ha accesso alla catena del Reasoner.

        Parametri:
            - claim: testo completo del claim originale
            - causality_type: tipo di causalità individuata dal reasoner
            - warrant_info: informazioni sul warrant (articolo/statuto) collegato
            - attacking_descriptions: descrizioni delle causalità da cui partire
            - include_precedents: se includere precedenti nella catena argomentativa
            - max_statutes: numero massimo di statuti da menzionare
            - max_precedents: numero massimo di precedenti da menzionare
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
    Sei un assistente legale esperto. Il tuo compito è costruire una catena logica
    che smonti il seguente claim legale, basandoti esclusivamente su:

    1. Il claim originale:
    \"\"\"{claim}\"\"\"

    2. La causalità identificata dal Reasoner:
    {causality_type}

    3. Il warrant collegato al claim:
    - Statuto/Articolo: {warrant_info.get('warrant', {}).get('denominazione', 'N/A')}
    - Riferimento: {warrant_info.get('warrant', {}).get('riferimento', '')}

    4. Descrizioni delle causalità attaccanti da considerare:
    - {attacking_text}

    5. Statuti rilevanti (max {max_statutes}) e precedenti rilevanti (max {max_precedents}):
    - Estrai dalle informazioni disponibili, assicurati di citare solo articoli o precedenti pertinenti
    - Se non ci sono statuti/precedenti rilevanti, spiega logicamente perché il claim può essere contro-argomentato senza di essi

    Istruzioni specifiche:
    - Genera una catena argomentativa chiara e sequenziale.
    - Ogni passaggio deve essere numerato.
    - Alla fine, produci una sintesi conclusiva che smonta il claim.
    - Se include precedenti, menziona nome, anno e breve sintesi del principio giuridico.
    - Non fare supposizioni non supportate da statuti o precedenti disponibili.
    - Mantieni il tono tecnico-legale, chiaro e conciso.

    Output atteso:
    1. Passaggi della catena logica numerati
    2. Sintesi finale conclusiva che smonta il claim
    """

        if include_precedents:
            prompt += "\nNota: includi i precedenti solo se supportano chiaramente la contro-argomentazione.\n"

        prompt += "\nGenera la catena logica ora:\n"

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
