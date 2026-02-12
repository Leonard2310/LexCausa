"""
LexCausa Counter-Reasoner Agent.

Generates independent counter-arguments, without using the Reasoner's chain.
Receives ONLY the claim + causal_type_id/theory_id from the Router and the
pre-retrieved contrary/neutral sources.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .aspic_formatter import AspicFormatter
from .base import AgentConfig, BaseAgent
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_statute_by_article_tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq  # noqa: E402


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
- **Catena di ragionamento**: followed by a numbered list (1. 2. 3. ...). This section is MANDATORY.
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
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
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
        Also adds articles that reference the current articles (reverse lookup from KB).
        """
        from .tools.neo4j_tools import _search_statutes_fallback
        
        try:
            import re
        except Exception:
            return statutes

        seen = {(s.get("articolo"), s.get("source")) for s in statutes}
        extra: List[dict] = []

        pattern = re.compile(r"art\.?\s*(\d{2,4})", re.IGNORECASE)

        for s in statutes:
            text = s.get("testo") or ""
            source = s.get("source", "codice_civile")
            articolo = str(s.get("articolo", "")).replace("art", "").strip()
            
            refs = set(pattern.findall(text))
            for token in re.findall(r"\b(\d{2,4})\b", text):
                if "/" in token:
                    continue
                refs.add(token)

            # Dynamic lookup from KB: find articles that mention this one
            # Uses existing fulltext search function
            try:
                reverse_results = _search_statutes_fallback(
                    query=f"art. {articolo}",
                    codice=source,
                    libro=None,
                    limit=5,
                )
                for r in reverse_results:
                    if not r.get("error") and r.get("articolo"):
                        ref_art = str(r["articolo"]).replace("art", "").strip()
                        if ref_art != articolo:  # Exclude self
                            refs.add(ref_art)
            except Exception:
                pass  # Fallback search failed, continue with forward refs only

            added_refs: List[str] = []
            for ref in refs:
                key = (ref, source)
                if key in seen:
                    continue
                seen.add(key)
                result = get_statute_by_article_tool.invoke(
                    {"articolo": ref, "codice": source}
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

    def _build_react_agent(self, api_key: str, model: str):
        """Build a fresh ReAct agent with specified key and model (for resilient invocation)."""
        llm = get_chat_groq(
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=api_key,
        )
        return create_react_agent(
            llm,
            self.tools,
            prompt=COUNTER_REASONER_SYSTEM_PROMPT,
        )

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

        # ----------------------------------------------------------
        # Execute with iterative step-by-step chain generation
        # ----------------------------------------------------------
        MAX_CHAIN_RETRIES = settings.chain_max_retries
        output = None

        for attempt in range(1, MAX_CHAIN_RETRIES + 1):
            self._log(
                f"🔄 Counter-Reasoner generation attempt {attempt}/{MAX_CHAIN_RETRIES}"
            )

            raw_output, iterative_chain = self._generate_counter_chain_iteratively(
                claim=claim,
                routing_decision=routing_decision,
                attack_selection=attack_selection,
                knowledge_base=knowledge_base,
                allowed_statutes=allowed_statutes,
                allowed_precedents=allowed_precedents,
            )

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

            # Use iterative chain directly; fall back to extraction if empty
            output.reasoning_chain = (
                iterative_chain
                if iterative_chain
                else self._extract_reasoning_chain(raw_output)
            )
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

            # Validate: ASPIC_IR must contain reasoning chain nodes (S1, S2, …)
            if self._has_valid_reasoning_chain(output.aspic_ir):
                chain_len = len(output.aspic_ir.get("reasoning_chain", []))
                self._log(
                    f"✅ Valid reasoning chain ({chain_len} steps) on attempt {attempt}",
                    "success",
                )
                break
            else:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: empty reasoning chain "
                    f"(no S* nodes in ASPIC_IR) — retrying…",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    self._log(
                        f"❌ Failed to generate a valid reasoning chain after "
                        f"{MAX_CHAIN_RETRIES} attempts",
                        "error",
                    )

        assert (
            output is not None
        ), "CounterReasonerOutput was never assigned"  # guard for mypy
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

    def _extract_cited_articles(self, text: str) -> List[str]:
        """Extract article references cited in the text."""
        import re

        pattern = re.compile(
            r"(?:art(?:icolo)?\.?\s*)(\d{1,4})\s*"
            r"(c\.?[cp]\.?|cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?))",
            re.IGNORECASE,
        )
        matches = pattern.findall(text)
        articles: List[str] = []
        for num, code in matches:
            code_norm = (
                "c.c." if "c" in code.lower() and "p" not in code.lower() else "c.p."
            )
            articles.append(f"Art. {num} {code_norm}")
        seen: set[str] = set()
        unique: List[str] = []
        for a in articles:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    def _generate_counter_chain_iteratively(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_selection: AttackSelection,
        knowledge_base: str,
        allowed_statutes: List[str],
        allowed_precedents: List[str],
    ) -> tuple[str, List[str]]:
        """Generate counter-reasoning chain step-by-step.

        Same iterative approach as Reasoner but with counter-argument
        framing.  The LLM autonomously decides when to stop; only
        ``chain_max_steps`` acts as a safety cap.

        Returns
        -------
        (raw_response, reasoning_chain)
        """
        MAX_STEPS = settings.chain_max_steps
        MIN_STEPS = settings.chain_min_steps
        steps: List[str] = []
        used_norms: List[str] = []

        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )

        attack_id = attack_selection.attack_id or "N/A"
        attack_desc = attack_selection.description or "N/A"

        for step_num in range(1, MAX_STEPS + 1):
            is_last_possible = step_num == MAX_STEPS
            can_conclude = step_num >= MIN_STEPS

            # --- EVALUATION PHASE (separate LLM call) ---
            if can_conclude and not is_last_possible and steps:
                should_continue = self._evaluate_should_continue(
                    claim=claim,
                    domain=routing_decision.domain,
                    steps=steps,
                    used_norms=used_norms,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    role="counter",
                )
                if not should_continue:
                    self._log(
                        f"🏁 Counter-chain concluded at step {step_num - 1} "
                        f"by evaluator (before generating step {step_num})"
                    )
                    break

            if steps:
                prev_context = "\n".join(
                    f"  Step {i + 1}: {s}" for i, s in enumerate(steps)
                )
                used_norms_text = ", ".join(used_norms) if used_norms else "none"
            else:
                prev_context = "  (No previous steps — you are at step 1.)"
                used_norms_text = "none"

            # ---- Per-step prompt (English, response in Italian) ----
            last_step_notice = (
                "\nTHIS IS THE LAST ALLOWED STEP. "
                "You MUST conclude with a STRONG statement AGAINST the claim. "
                "Your conclusion must clearly state why the claim FAILS or is LEGALLY UNFOUNDED."
                if is_last_possible
                else ""
            )
            step_prompt = f"""You are an expert Italian jurist acting as PROSECUTOR/OPPOSING COUNSEL. You are building a COUNTER-ARGUMENT STEP BY STEP to DISMANTLE and REJECT the following legal claim.

YOUR STANCE: You are the ADVERSARY of the claim. You MUST argue that the claim is LEGALLY UNFOUNDED.
Every step must provide ONE concrete reason WHY the claim FAILS under Italian law.
NEVER mention strengths, merits, or valid points of the claim.
Do NOT balance pros and cons. You are EXCLUSIVELY anti-claim.

CLAIM TO ATTACK (you must DEMOLISH this):
"{claim}"

ATTACK STRATEGY: {attack_id}
DESCRIPTION: {attack_desc}

DOMAIN: {routing_decision.domain}
CAUSAL TYPE: {routing_decision.causal_type_id}
THEORY: {routing_decision.theory_id}

=== KNOWLEDGE BASE (use ONLY these sources) ===
{knowledge_base}
=== END KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
{statutes_list}

ALLOWED PRECEDENT REFERENCES (do not cite others):
{precedents_list}

--- CURRENT COUNTER-ARGUMENT STATE ---
Steps completed so far:
{prev_context}

Norms already used: {used_norms_text}
Current step: {step_num} (safety cap: {MAX_STEPS})

--- INSTRUCTIONS FOR STEP {step_num} ---
Generate EXACTLY ONE ATOMIC counter-reasoning step (step {step_num}).

ATOMIC STEP RULES:
- This step is ONE SMALL PIECE of a multi-step counter-argument chain. Do NOT try to give a complete rebuttal.
- Focus on EXACTLY ONE weak point, one norm violation, or one factual gap. Do NOT cover multiple issues.
- 2-4 sentences MAXIMUM. Be concise and precise.
- Do NOT repeat or summarize the claim. The claim is already known.
- Do NOT restate conclusions from previous steps. Build on them.

ANTI-CLAIM REQUIREMENTS for this step:
1. ONE ATTACK POINT ONLY: Pick exactly ONE of the following for this step:
   - Show how ONE specific norm CONTRADICTS or LIMITS the claim, OR
   - Expose ONE legal prerequisite that is NOT satisfied, OR
   - Draw ONE narrow conclusion showing the claim FAILS on a specific point
2. NORM COVERAGE: Try to cite an article NOT yet used ({used_norms_text}).
   You MAY reuse an already-cited article ONLY if you apply it to a DIFFERENT factual aspect
   that was NOT discussed in any previous step. Never repeat the same reasoning.
   If you have nothing new to add (no new aspect, no new norm), respond with STEP: DONE.
3. ATTACK ANGLE: Use the strategy "{attack_id}" as the main lens
4. CONNECT to the previous step: your step must start from where the last step ended.
   If step N-1 exposed flaw X, step N should use X to attack further.
5. ALWAYS DISFAVOR THE CLAIM: interpret norms and facts in the way most unfavorable to the claimant.
{last_step_notice}

RESPONSE FORMAT:
STEP: [Your atomic counter-reasoning step in Italian — max 4 sentences]

CRITICAL RULES:
- Your ENTIRE STEP text must be written in Italian.
- MAX 4 sentences. If you need more, you are covering too much — split it.
- Cite exactly one specific article (e.g. Art. 52 c.p.).
- FACTUAL FIDELITY: Use ONLY facts explicitly stated in the CLAIM above. Do NOT add, infer, assume, or invent facts that are not written in the claim. For example, if the claim says ONE strike, do NOT say multiple strikes. If the claim does not mention a detail, do NOT fabricate it.
- Do NOT mention the Reasoner. Produce a standalone counter-argument.
- Do NOT invent sources not present in the knowledge base.
- NEVER argue in favor of the claim. You are the OPPOSITION.
- Do NOT write a complete rebuttal. Write ONE building block of the counter-argument.
- NEVER write anything that supports or validates the claim. Every sentence must ATTACK it.
"""

            self._log(f"🔗 Generating counter-step {step_num}/{MAX_STEPS}...")

            try:
                resp = self._resilient_llm_invoke([HumanMessage(content=step_prompt)])
                step_response = (resp.content or "").strip()
            except Exception as e:
                self._log(
                    f"⚠️ Counter-step {step_num} generation failed: {e}", "warning"
                )
                break

            step_text = self._parse_step_text(step_response)

            if not step_text or step_text.strip().upper() == "DONE":
                self._log(
                    f"⚠️ Counter-step {step_num}: no new norm available, stopping",
                    "warning",
                )
                break

            # --- REPETITION DETECTION (programmatic) ---
            if steps and self._is_repetitive_step(step_text, steps):
                self._log(
                    f"🔁 Counter-step {step_num}: too similar to a previous step, "
                    f"stopping chain (repetition detected)"
                )
                break

            steps.append(step_text)

            new_norms = self._extract_cited_articles(step_text)
            used_norms.extend(new_norms)

            self._log(
                f"✅ Counter-step {step_num}: {step_text[:80]}... "
                f"| norms: {', '.join(new_norms) if new_norms else 'none'}"
            )

            # Last possible step: forced stop
            if is_last_possible:
                self._log(f"🏁 Counter-chain stopped at safety cap (step {step_num})")
                break

        if not steps:
            self._log("❌ No steps generated in iterative counter-chain", "error")
            return "", []

        self._log(
            f"📊 Iterative counter-chain complete: {len(steps)} steps, "
            f"{len(set(used_norms))} unique norms"
        )

        raw_response = self._assemble_counter_raw_response(claim, steps, attack_id)
        return raw_response, steps

    def _parse_step_text(self, response: str) -> str:
        """Extract the step text from an LLM response.

        Looks for a ``STEP:`` / ``STEP N:`` (or ``PASSO:``) marker and
        returns everything after it.  Falls back to the full response
        if no marker is found.
        
        Also removes any leading numeric prefixes (e.g., "3 Per valutare...")
        that the LLM may have included.
        """
        lines = response.strip().split("\n")
        step_lines: List[str] = []
        found_step = False

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            # Match STEP:, STEP 1:, STEP 12:, PASSO:, PASSO 1: etc.
            if re.match(r"^(STEP|PASSO)\s*\d*\s*:", upper):
                content = re.sub(
                    r"^(?:STEP|PASSO)\s*\d*\s*:\s*",
                    "",
                    stripped,
                    flags=re.IGNORECASE,
                ).strip()
                if content:
                    step_lines.append(content)
                found_step = True
                continue

            if found_step and stripped:
                step_lines.append(stripped)

        step_text = " ".join(step_lines).strip()

        # Fallback: use entire response (skip DECISION / STEP N: prefixes)
        if not step_text:
            fallback_lines = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                su = s.upper()
                if su.startswith("DECISION:") or su.startswith("DECISIONE:"):
                    continue
                # Strip any leading STEP N: prefix
                s = re.sub(
                    r"^(?:STEP|PASSO)\s*\d*\s*:\s*",
                    "",
                    s,
                    flags=re.IGNORECASE,
                )
                if s:
                    fallback_lines.append(s)
            step_text = " ".join(fallback_lines).strip()

        # P6 FIX: Remove leading numeric prefixes like "3 Per valutare..."
        # This handles cases where LLM responds with just "3 Per valutare" without STEP:
        step_text = re.sub(r"^\d+\s+", "", step_text)
        
        return step_text

    def _evaluate_should_continue(
        self,
        claim: str,
        domain: str,
        steps: List[str],
        used_norms: List[str],
        knowledge_base: str,
        statutes_list: str,
        role: str = "counter",
    ) -> bool:
        """Dedicated evaluation call: should the chain continue?

        A lightweight LLM call that inspects the current chain and
        decides whether an additional step would meaningfully
        strengthen the argument.

        Returns ``True`` to continue, ``False`` to conclude.
        """
        prev_context = "\n".join(
            f"  Step {i + 1}: {s[:300]}..." for i, s in enumerate(steps)
        )
        used_text = ", ".join(sorted(set(used_norms))) if used_norms else "none"
        n_steps = len(steps)
        n_unique_norms = len(set(used_norms)) if used_norms else 0

        role_desc = "supporting" if role == "support" else "counter-"

        eval_prompt = f"""You are a senior Italian jurist evaluating whether a legal argument needs more steps.

A {role_desc}argument for the claim below has {n_steps} steps so far.
It cites {n_unique_norms} unique norms out of {len(used_norms)} total citations.

CLAIM: "{claim}"
DOMAIN: {domain}

Steps so far:
{prev_context}

Norms already cited: {used_text}

AVAILABLE STATUTES (not yet used):
{statutes_list}

EVALUATION GUIDELINES:
- A well-constructed legal argument typically needs 3-6 distinct steps covering
  different legal aspects or applying norms to different facts.
- With only {n_steps} step(s), consider whether there are still pertinent aspects to cover.
- Each step should add NEW reasoning: a new norm, or a new application of an existing norm to a different fact.

CRITICAL — REPETITION DETECTION:
- REPETITION = two steps make the SAME legal point about the SAME factual aspect. This is BAD.
- GOOD COVERAGE = each step addresses a DIFFERENT legal aspect or applies a norm to a DIFFERENT fact.
- Re-citing an article is OK if applied to a genuinely different aspect of the case.
- If the last step simply rephrases or restates what a previous step already said, answer CONCLUDE.

DECISION RULES:
- Answer CONTINUE if ALL of these are true:
  (1) each step so far addresses a DIFFERENT legal aspect or fact (no repetition), AND
  (2) there is at least one pertinent unused norm OR a new factual aspect to cover, AND
  (3) fewer than 6 steps have been generated.
- Answer CONCLUDE if ANY of these is true:
  (a) any two steps make the SAME legal point about the SAME fact (repetition), OR
  (b) all pertinent legal aspects have been covered, OR
  (c) 6+ steps have already been generated.

YOUR ANSWER (exactly one word — CONTINUE or CONCLUDE):"""

        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=eval_prompt)])
            answer = (resp.content or "").strip().upper()
        except Exception as e:
            self._log(
                f"⚠️ Evaluation call failed: {e}; defaulting to CONTINUE",
                "warning",
            )
            return True

        # Robust parsing
        if "CONCLUD" in answer:
            should_continue = False
        elif "CONTINU" in answer:
            should_continue = True
        else:
            # Ambiguous → default to CONCLUDE after enough steps
            should_continue = n_steps < 5
            self._log(
                f"⚠️ Evaluator ambiguous: '{answer[:50]}'; "
                f"defaulting to {'CONTINUE' if should_continue else 'CONCLUDE'}",
                "warning",
            )

        self._log(
            f"🔍 Evaluator: {'CONTINUE' if should_continue else 'CONCLUDE'} "
            f"(raw: '{answer[:40]}')"
        )
        return should_continue

    def _assemble_counter_raw_response(
        self,
        claim: str,
        steps: List[str],
        attack_id: str,
    ) -> str:
        """Assemble counter-argument raw response from iterative steps.
        
        Ensures the conclusion is framed as OPPOSING the claim, not supporting it.
        """
        chain_section = "**Catena di ragionamento**:\n"
        for i, step in enumerate(steps, 1):
            chain_section += f"{i}. {step}\n"

        premise_steps = steps[:-1] if len(steps) > 1 else steps
        conclusion_step = steps[-1] if steps else ""

        premise_text = " ".join(premise_steps)
        norms = self._extract_cited_articles(" ".join(steps))
        norms_text = "\n".join(f"- {n}" for n in norms) if norms else "N/D"

        # Build attack-specific causal link description
        attack_descriptions_it = {
            "but_for_fails": "il nesso causale controfattuale fallisce: l'evento si sarebbe verificato comunque",
            "no_covering_law_or_low_support": "manca una legge di copertura o il supporto probabilistico è insufficiente",
            "alternative_causal_path": "esiste un percorso causale alternativo plausibile",
            "duty_to_act_missing_for_omission": "per l'omissione, manca l'obbligo giuridico di agire",
            "abnormal_or_atypical_chain": "la catena causale è anormale/atipica e interrompe l'imputabilità",
            "sole_sufficient_cause": "una causa sopravvenuta era da sola sufficiente e interrompe il nesso",
            "intervening_cause_breaks_chain": "un fattore autonomo sopravvenuto interrompe la catena",
            "force_majeure_filter": "caso fortuito/forza maggiore esclude l'imputabilità",
            "damage_is_indirect": "il danno è indiretto o mediato rispetto al fatto base",
            "damage_not_foreseeable": "il danno non era prevedibile ex ante",
            "creditor_contributed": "il creditore ha concorso a causare l'evento/danno",
            "creditor_failed_to_mitigate": "il creditore non ha mitigato il danno evitabile",
            "quantification_uncertain": "la quantificazione del danno è incerta o speculativa",
            "event_was_avoidable": "l'evento era evitabile con l'ordinaria diligenza",
            "event_was_foreseeable": "l'evento era prevedibile e non fortuito",
            "risk_was_assumed_or_controllable": "il rischio era assunto o controllabile",
        }
        attack_desc_it = attack_descriptions_it.get(
            attack_id, 
            "le norme citate indeboliscono il fondamento giuridico della pretesa"
        )

        # Ensure the conclusion opposes the claim
        conclusion_prefix = "Pertanto, la pretesa deve essere RIGETTATA poiché "
        
        raw = (
            f"**Premessa Alternativa**: {premise_text}\n\n"
            f"**Norma**:\n{norms_text}\n\n"
            f"**Nesso Causale Alternativo**: L'analisi giuridica dimostra che "
            f"{attack_desc_it}. La catena argomentativa evidenzia come "
            f"le norme applicabili al caso non supportino la pretesa del ricorrente, "
            f"ma anzi ne rivelino l'infondatezza giuridica.\n\n"
            f"**Conclusione Contraria**: {conclusion_prefix}{conclusion_step}\n\n"
            f"{chain_section}"
        )
        return raw

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
   **Premessa Alternativa**: (incompatible with the claim — write in prose, NO numbered lists)
   **Norma**: (cite MULTIPLE relevant statutes from ALLOWED STATUTES, not just one;
              if none apply, omit this section — use bullet points with "-" if listing multiple norms, NEVER numbered lists)
   **Nesso Causale Alternativo**: (write in prose, NO numbered lists)
   **Conclusione Contraria**: (write in prose, NO numbered lists)
3) After the counter-arguments, you MUST add the following header and numbered chain.
   This section is MANDATORY and must NEVER be omitted:

   **Catena di ragionamento**:
   1. [First reasoning step — cite the specific article(s) it relies on, e.g. Art. XX c.p.]
   2. [Second reasoning step — cite the specific article(s)]
   3. [Continue for each logical step...]

   RULES for the numbered chain:
   - Use EXACTLY the header "**Catena di ragionamento**:" before the numbered list.
   - Each step MUST be on its own line, starting with "N. " (e.g. "1. ", "2. ", "3. ").
   - Each step MUST reference at least one specific article (e.g. "Art. 40 c.p.").
   - The chain must have AT LEAST 3 numbered steps.
   - Do NOT mention the Reasoner.
   - If you omit this section, your response is INVALID.

FORMATTING RULE — CRITICAL:
- Numbered lists ("1. ", "2. ", "3. ", etc.) are ONLY allowed inside the **Catena di ragionamento** section.
- In ALL other sections (Premessa Alternativa, Norma, Nesso Causale Alternativo, Conclusione Contraria),
  use ONLY prose text or bullet points with "-". NEVER use numbered lists outside the chain.

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
