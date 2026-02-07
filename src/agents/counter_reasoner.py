"""
LexCausa Counter-Reasoner Agent.

Generates independent counter-arguments, without using the Reasoner's chain.
Receives ONLY the claim + causal_type_id/theory_id from the Router and the
pre-retrieved contrary/neutral sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .aspic_formatter import AspicFormatter
from .base import AgentConfig, BaseAgent
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_statute_by_article_tool


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
    causal_type_id: str
    theory_id: str
    selected_attack_id: str = ""
    reasoner_causality: dict = field(default_factory=dict)  # compatibility alias
    relevant_statutes: List[dict] = field(default_factory=list)
    relevant_precedents: List[dict] = field(default_factory=list)
    counter_arguments: List[CounterArgument] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    raw_response: str = ""
    aspic_ir: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causal_type_id": self.causal_type_id,
            "theory_id": self.theory_id,
            "selected_attack_id": self.selected_attack_id,
            "reasoner_causality": self.reasoner_causality,
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
            "aspic_ir": self.aspic_ir,
        }


# System prompt for the Counter-Reasoner (with pre-retrieved context)
COUNTER_REASONER_SYSTEM_PROMPT = """IMPORTANT: You MUST respond ENTIRELY in Italian. Every word of your response must be in Italian.

You are the Counter-Reasoner. Dismantle the claim independently.
You receive:
- causal_type_id and theory_id fixed by the Router (do not re-classify)
- selected_attack_id chosen from the config attack pool
- KNOWLEDGE BASE with contrary/neutral statutes and precedents

Critical rules:
- Use ONLY the sources in the KNOWLEDGE BASE; do not invent statutes or precedents.
- Do not reference the Reasoner or its reasoning; produce a standalone counter-argument.
- If a helpful statute is missing from the knowledge base, omit the citation instead of inventing it.
- Always cite the statute number/code when available (e.g., "Art. 41 c.p.").

Expected structure (use these EXACT Italian headers):
- **Premessa Alternativa**
- **Norma** (only if present in ALLOWED STATUTES)
- **Nesso Causale Alternativo**
- **Conclusione Contraria**
- Numbered counter-reasoning chain.
MANDATORY: Your ENTIRE response must be written in Italian. Do NOT write in English."""


ATTACK_DESCRIPTIONS: Dict[str, str] = {
    "but_for_fails": "Counterfactual fails: the event would have occurred anyway.",
    "no_covering_law_or_low_support": "Missing covering law or insufficient support for probabilistic counterfactual.",
    "alternative_causal_path": "A plausible alternative causal path exists.",
    "duty_to_act_missing_for_omission": "For omission, the legal duty to act is missing.",
    "abnormal_or_atypical_chain": "The causal chain is abnormal/atypical and breaks imputability.",
    "sole_sufficient_cause": "A supervening cause alone was sufficient and breaks the link.",
    "intervening_cause_breaks_chain": "An intervening autonomous factor breaks the chain.",
    "force_majeure_filter": "Fortuitous event/force majeure excludes imputability.",
    "damage_is_indirect": "The damage is indirect or mediated relative to the base fact.",
    "damage_not_foreseeable": "The damage was not foreseeable ex ante (e.g., art. 1225 c.c.).",
    "creditor_contributed": "The creditor contributed to causing the event/damage.",
    "creditor_failed_to_mitigate": "The creditor failed to mitigate avoidable damage (art. 1227 c.c.).",
    "quantification_uncertain": "Damage quantification is uncertain or speculative.",
    "event_was_avoidable": "The event was avoidable with ordinary diligence.",
    "event_was_foreseeable": "The event was foreseeable; it is not fortuitous.",
    "risk_was_assumed_or_controllable": "The risk was assumed or controllable, so not fortuitous.",
}


class CounterReasoner(BaseAgent):
    """
    Legal Counter-Reasoner Agent.

    Generates counter-arguments to challenge the claim independently of the Reasoner.
    Selects the attack from config (counter_attack_pool) based on causal_type_id/theory_id.

    Flow:
    1. api_server pre-retrieves statutes and precedents
    2. CounterReasoner.run() receives the Router decision + pre-retrieved knowledge
    3. ReAct agent builds counter-arguments using only contrary/neutral sources
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Counter-Reasoner agent."""
        super().__init__(config)
        self._react_agent = None
        self._config = config_loader.load_config()

    # ------------------------------------------------------------------
    # Attack selection logic (config-driven)
    # ------------------------------------------------------------------
    def _select_attack(
        self, claim: str, routing_decision: RoutingDecision
    ) -> AttackSelection:
        """Select counter attack id based on config pools."""
        pool: List[str] = config_loader.counter_attack_pool_for(
            routing_decision.causal_type_id, self._config
        )
        theory_attacks: List[str] = config_loader.theory_counter_attacks(
            routing_decision.theory_id, self._config
        )

        if theory_attacks:
            intersection = [a for a in pool if a in theory_attacks]
            if intersection:
                pool = intersection

        if not pool:
            # Fallback to theory attacks or all known attacks
            pool = theory_attacks or list(ATTACK_DESCRIPTIONS.keys())

        attack_id = self._pick_attack_with_llm(
            claim, routing_decision.causal_type_id, routing_decision.theory_id, pool
        )

        return AttackSelection(
            pool=pool,
            attack_id=attack_id,
            description=ATTACK_DESCRIPTIONS.get(attack_id, ""),
        )

    def _pick_attack_with_llm(
        self,
        claim: str,
        causal_type_id: str,
        theory_id: str,
        pool: List[str],
    ) -> str:
        """Use LLM to pick the most suitable attack id from pool."""
        if not pool:
            return ""

        options_text = "\n".join(
            f"- {aid}: {ATTACK_DESCRIPTIONS.get(aid, '')}" for aid in pool
        )
        prompt = f"""Claim:
"{claim}"

Routing context:
- causal_type_id: {causal_type_id}
- theory_id: {theory_id}

Select the most useful attack among the following ids and return ONLY the chosen id:
{options_text}
"""
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip()
            attack_id = self._clean_attack_choice(answer, pool)
            if attack_id:
                return attack_id
        except Exception as e:
            self._log(f"⚠️ LLM attack selection failed: {e}", "warning")

        return pool[0]

    def _clean_attack_choice(self, raw: str, pool: List[str]) -> str:
        """Normalize LLM output to a valid attack id."""
        candidate = raw.replace("`", "").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        candidate = candidate.split()[0] if candidate else ""
        for attack_id in pool:
            if attack_id.lower() in candidate.lower():
                return attack_id
        return ""

    def _expand_with_cross_references(self, statutes: List[dict]) -> List[dict]:
        """
        Add statutes referenced inside article text (one hop) to avoid missing rinvii.
        """
        try:
            import re
        except Exception:
            return statutes

        seen = {(s.get("articolo"), s.get("source")) for s in statutes}
        extra: List[dict] = []

        pattern = re.compile(r"art\.?\s*(\d{2,4})", re.IGNORECASE)

        for s in statutes:
            text = s.get("testo") or ""
            refs = set(pattern.findall(text))
            for token in re.findall(r"\b(\d{2,4})\b", text):
                if "/" in token:
                    continue
                refs.add(token)

            added_refs: List[str] = []
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

        merged = statutes + extra
        deduped: List[dict] = []
        seen_final = set()
        for st in merged:
            k = (st.get("articolo"), st.get("source"))
            if k in seen_final:
                continue
            seen_final.add(k)
            deduped.append(st)
        return deduped

    def _filtered_anchor_statutes_for_types(
        self, causal_types: List[str], claim: str
    ) -> List[dict]:
        """
        Get taxonomy anchor norms for the given causal types.

        Reads directly from config_taxonomy.json using the correct keys:
        - anchor_norms -> core_norms / accessory_norms
        - Each norm has 'ref' and 'role' fields
        """
        statutes: List[dict] = []
        seen_refs = set()
        unique_cts = list(dict.fromkeys(ct for ct in causal_types if ct))

        for ct in unique_cts:
            # Find causal_type block in config
            causal_type_block = next(
                (c for c in self._config.get("causal_types", []) if c.get("id") == ct),
                None,
            )
            if not causal_type_block:
                self._log(
                    f"⚠️ [taxonomy] Causal type {ct} not found in config", "warning"
                )
                continue

            anchor_norms = causal_type_block.get("anchor_norms", {})
            core_norms = anchor_norms.get("core_norms", []) or []
            accessory_norms = anchor_norms.get("accessory_norms", []) or []
            all_norms = core_norms + accessory_norms

            self._log(
                f"🔎 [taxonomy] Causality {ct}: core {len(core_norms)}/{len(core_norms)}, accessory {len(accessory_norms)}/{len(accessory_norms)}"
            )

            kept_refs = []
            for n in all_norms:
                ref = n.get("ref")
                # role = n.get("role", "")
                """
                TODO: Valutarne l'inserimento futuro per rendere la contro-argomentazione più sofisticata
                (ad esempio, distinguendo tra attacchi alle norme core e accessorie).
                """
                if not ref or ref in seen_refs:
                    continue
                seen_refs.add(ref)
                kept_refs.append(ref)
                # Convert to statute dict format
                statutes.append(self._norm_to_statute_dict(n))

            if kept_refs:
                self._log(f"   ✔️ Kept: {', '.join(kept_refs)}")

        return statutes

    @property
    def tools(self) -> list:
        """
        Get the tools available to this agent.

        NOTE: No search tools - the agent works with pre-retrieved context.
        Only statute lookup to keep independence from Reasoner.
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
                prompt=COUNTER_REASONER_SYSTEM_PROMPT,
            )
        return self._react_agent

    def run(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        pre_retrieved_statutes: List[dict],
        pre_retrieved_precedents: List[dict],
    ) -> CounterReasonerOutput:
        """
        Execute the counter-reasoning process with pre-retrieved knowledge.

        Args:
            claim: The legal claim to counter-argue.
            routing_decision: Output of the Router with causal_type_id/theory_id.
            pre_retrieved_statutes: Already retrieved and filtered statute articles.
            pre_retrieved_precedents: Already retrieved precedents.

        Returns:
            CounterReasonerOutput with counter-arguments and reasoning chain.
        """
        self._log(f"Counter-analyzing claim: {claim[:100]}...")
        self._log(
            f"📚 Knowledge base: {len(pre_retrieved_statutes)} statutes, {len(pre_retrieved_precedents)} precedents"
        )

        if not routing_decision or not routing_decision.causal_type_id:
            raise ValueError(
                "routing_decision with causal_type_id/theory_id is required"
            )

        # Select counter attack from config pools
        attack_selection = self._select_attack(claim, routing_decision)
        self._log(
            f"⚔️ Selected counter attack: {attack_selection.attack_id or 'N/A'} "
            f"(pool size {len(attack_selection.pool)})"
        )
        self._log(f"📝 Attack description: {attack_selection.description or 'N/A'}")

        all_statutes = pre_retrieved_statutes
        # Add filtered anchor norms for provided causal types (router + additional)
        anchor_statutes = self._filtered_anchor_statutes_for_types(
            [routing_decision.causal_type_id]
            + (routing_decision.additional_causal_types or []),
            claim,
        )
        if anchor_statutes:
            self._log(
                f"🧭 Anchor norms added to KB (counter): {len(anchor_statutes)}",
                "info",
            )
        all_statutes = all_statutes + anchor_statutes
        # Deduplicate
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

        # Format knowledge base for prompt
        knowledge_base = self._format_context_for_prompt(
            deduped_statutes, pre_retrieved_precedents
        )

        allowed_statutes = [
            f"Art. {s.get('articolo')} ({'c.c.' if s.get('source') == 'codice_civile' else 'c.p.'})"
            for s in deduped_statutes
        ]
        allowed_precedents = [
            p.get("title", "Untitled") for p in pre_retrieved_precedents
        ]

        # Build prompt with context
        input_prompt = self._build_counter_reasoning_prompt_with_context(
            claim=claim,
            routing_decision=routing_decision,
            attack_selection=attack_selection,
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
            # Handle Groq tool_use_failed: extract valid response from failed_generation
            recovered = self._extract_failed_generation(e)
            if recovered:
                self._log(
                    "⚠️ Tool call failed but valid response recovered from failed_generation",
                    "warning",
                )
                messages_out = []
                # Skip normal message extraction, use recovered text directly
                output = CounterReasonerOutput(
                    claim=claim,
                    causal_type_id=routing_decision.causal_type_id,
                    theory_id=routing_decision.theory_id,
                    selected_attack_id=attack_selection.attack_id,
                    reasoner_causality={
                        "causal_type_id": routing_decision.causal_type_id,
                        "theory_id": routing_decision.theory_id,
                    },
                    relevant_statutes=deduped_statutes,
                    relevant_precedents=pre_retrieved_precedents,
                    raw_response=recovered,
                )
                output.reasoning_chain = self._extract_reasoning_chain(recovered)
                output.counter_arguments = self._extract_arguments(recovered)
                output.reasoning_chain = self._sanitize_reasoning_chain(
                    output.reasoning_chain, pre_retrieved_precedents
                )
                formatter = AspicFormatter(
                    role="counter",
                    statutes=deduped_statutes,
                    precedents=pre_retrieved_precedents,
                )
                output.aspic_ir = formatter.format(
                    claim=claim,
                    raw_response=recovered,
                    reasoning_chain=output.reasoning_chain,
                    arguments=output.counter_arguments,
                    metadata={
                        "selected_attack_id": attack_selection.attack_id,
                        "causal_type_id": routing_decision.causal_type_id,
                        "theory_id": routing_decision.theory_id,
                    },
                )
                self._log(
                    f"✅ Generated {len(output.counter_arguments)} counter-arguments (recovered)",
                    "success",
                )
                return output

            # Graceful fallback for other errors
            error_msg = f"Errore durante l'esecuzione del Counter-Reasoner: {e}"
            self._log(error_msg, "error")
            return CounterReasonerOutput(
                claim=claim,
                causal_type_id=routing_decision.causal_type_id,
                theory_id=routing_decision.theory_id,
                selected_attack_id=attack_selection.attack_id,
                reasoner_causality={
                    "causal_type_id": routing_decision.causal_type_id,
                    "theory_id": routing_decision.theory_id,
                },
                relevant_statutes=pre_retrieved_statutes,
                relevant_precedents=pre_retrieved_precedents,
                counter_arguments=[],
                reasoning_chain=[error_msg],
                raw_response=error_msg,
            )

        # Log tool calls
        tool_names: list[str] = []
        for msg in messages_out:
            if hasattr(msg, "name") and msg.name:
                tool_names.append(msg.name)

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
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            selected_attack_id=attack_selection.attack_id,
            reasoner_causality={
                "causal_type_id": routing_decision.causal_type_id,
                "theory_id": routing_decision.theory_id,
            },
            relevant_statutes=deduped_statutes,
            relevant_precedents=pre_retrieved_precedents,
            raw_response=raw_output,
        )

        # Parse response
        output.reasoning_chain = self._extract_reasoning_chain(raw_output)
        output.counter_arguments = self._extract_arguments(raw_output)
        output.reasoning_chain = self._sanitize_reasoning_chain(
            output.reasoning_chain, pre_retrieved_precedents
        )

        formatter = AspicFormatter(
            role="counter",
            statutes=deduped_statutes,
            precedents=pre_retrieved_precedents,
        )
        output.aspic_ir = formatter.format(
            claim=claim,
            raw_response=raw_output,
            reasoning_chain=output.reasoning_chain,
            arguments=output.counter_arguments,
            metadata={
                "selected_attack_id": attack_selection.attack_id,
                "causal_type_id": routing_decision.causal_type_id,
                "theory_id": routing_decision.theory_id,
            },
        )

        self._log(
            f"✅ Generated {len(output.counter_arguments)} counter-arguments", "success"
        )
        return output

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
            end = error_str.find('"}}', start)
            if end == -1:
                end = error_str.find("'}", start)
            if end != -1:
                text = error_str[start:end]
                text = text.replace("\\n", "\n")
                if len(text) > 50:
                    return text
        return ""

    def _build_counter_reasoning_prompt_with_context(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_selection: AttackSelection,
        knowledge_base: str,
        allowed_statutes: List[str],
        allowed_precedents: List[str],
    ) -> str:
        """
        Build the prompt for CounterReasoner with pre-retrieved context.
        """
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )

        attack_id = attack_selection.attack_id or "N/A"
        attack_desc = attack_selection.description or "N/A"
        pool = attack_selection.pool
        pool_lines = "\n".join(f"- {a}" for a in pool) or "- No attack available"

        return f"""Analyze the claim and generate an independent counter-argument that dismantles the thesis.

CLAIM:
"{claim}"

ROUTING DECISION:
- causal_type_id: {routing_decision.causal_type_id}
- theory_id: {routing_decision.theory_id}

COUNTER ATTACK FOCUS (chosen from config):
- selected_attack_id: {attack_id}
- description: {attack_desc}
- candidate_pool:
{pool_lines}

=== KNOWLEDGE BASE (use ONLY these sources) ===
{knowledge_base}
=== END KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
{statutes_list}

ALLOWED PRECEDENT REFERENCES (do not cite others):
{precedents_list}

INSTRUCTIONS:
1) Use selected_attack_id as the main lens to attack the causal link.
2) Build one or more counter-arguments with EXACTLY this structure and these Italian headers:
   **Premessa Alternativa**: (incompatible with the claim)
   **Norma**: (cite MULTIPLE relevant statutes from ALLOWED STATUTES, not just one;
              if none apply, omit this section)
   **Nesso Causale Alternativo**:
   **Conclusione Contraria**:
3) End with a numbered counter-reasoning chain, without mentioning the Reasoner.
   Each step of the chain MUST reference the specific article(s) it relies on.

IMPORTANT - NORM USAGE REQUIREMENTS:
- You have {len(allowed_statutes)} statutes available. Cite EVERY article you deem pertinent
  to dismantling the claim — do not artificially limit yourself to a fixed number.
- Do NOT rely on a single norm for the entire counter-argument.
- For each factual aspect you attack (e.g., causation, foreseeability, duty, mitigation),
  identify the most specific applicable statute from the ALLOWED STATUTES list.
- Quote the relevant text from each statute when available in the KNOWLEDGE BASE.
- Anchor norms provide the framework, but integrate additional non-anchor statutes
  from the knowledge base that strengthen your counter-argument on the specific facts.
- COHERENCE RULE: Every norm you cite in the **Norma** section MUST appear in at least one
  step of the numbered counter-reasoning chain, with an explanation of its specific role.
  Do NOT list norms in **Norma** that you never use in the chain.

CRITICAL: Do not invent sources, do not mention the Reasoner.
MANDATORY LANGUAGE RULE: Your ENTIRE response MUST be written in Italian. Do NOT write in English. Every sentence, header, and explanation must be in Italian. Use EXACTLY the Italian headers shown above (Premessa Alternativa, Norma, Nesso Causale Alternativo, Conclusione Contraria)."""

    def _extract_arguments(self, response: str) -> List[CounterArgument]:
        """Estrae contro-argomenti strutturati dalla risposta."""
        arguments = []
        sections = response.split("**")
        current_arg: dict[str, str] = {}

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


@dataclass
class AttackSelection:
    pool: List[str]
    attack_id: str
    description: str
