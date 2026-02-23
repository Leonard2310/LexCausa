"""
LexCausa Counter-Reasoner Agent.

Generates counter-arguments on the same claim, optionally targeting the
Reasoner's conclusion when available.
Receives the claim + causal_type_id/theory_id from the Router and the
pre-retrieved relevant sources.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .aspic_formatter import AspicFormatter
from .base import AgentConfig, BaseAgent
from .citation_utils import (
    extract_article_mentions,
    format_article_citation,
    infer_source_hint,
    normalize_article_id,
)
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_statute_by_article_tool
from .tools.prompt_registry import get_prompt, render_prompt

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
    selected_attack_ids: List[str] = field(default_factory=list)
    reasoner_causality: dict = field(default_factory=dict)  # compatibility alias
    relevant_statutes: List[dict] = field(default_factory=list)
    relevant_precedents: List[dict] = field(default_factory=list)
    counter_arguments: List[CounterArgument] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    raw_response: str = ""
    aspic_ir: dict = field(default_factory=dict)
    abstained: bool = False
    abstention_reason: str = ""
    reasoner_conclusion_context: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causal_type_id": self.causal_type_id,
            "theory_id": self.theory_id,
            "selected_attack_id": self.selected_attack_id,
            "selected_attack_ids": self.selected_attack_ids,
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
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "reasoner_conclusion_context": self.reasoner_conclusion_context,
        }


# System prompt for the Counter-Reasoner (with pre-retrieved context)
COUNTER_REASONER_SYSTEM_PROMPT = get_prompt("counter_reasoner.system")


_DEFAULT_ATTACK_DESCRIPTION_EN = (
    "Counter-argument to weaken causal/legal support of the claim."
)
_DEFAULT_ATTACK_DESCRIPTION_IT = (
    "le norme citate indeboliscono la tesi giuridica principale"
)


class CounterReasoner(BaseAgent):
    """
    Legal Counter-Reasoner Agent.

    Generates counter-arguments against the primary legal thesis,
    using the Reasoner's conclusion when available.
    Selects multiple attacks from config (counter_attack_pool) based on causal_type_id/theory_id.

    Flow:
    1. api_server pre-retrieves statutes and precedents
    2. CounterReasoner.run() receives the Router decision + pre-retrieved knowledge
    3. ReAct agent builds counter-arguments using the retrieved relevant sources
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Counter-Reasoner agent."""
        super().__init__(config)
        self._react_agent = None
        self._config = config_loader.load_config()
        self._attack_descriptions_en = config_loader.counter_attack_descriptions(
            self._config,
            locale="en",
        )
        self._attack_descriptions_it = config_loader.counter_attack_descriptions(
            self._config,
            locale="it",
        )
        self._max_plan_retries = 3
        self._max_step_rewrites = 3
        self._reasoner_opposition_check_cache: Dict[
            tuple[str, str], tuple[bool, str]
        ] = {}

    def _known_attack_ids(self) -> List[str]:
        """Return all known attack IDs from taxonomy metadata."""
        ids: List[str] = []

        def _append_if_new(values: List[str]) -> None:
            for value in values:
                attack_id = str(value).strip()
                if attack_id and attack_id not in ids:
                    ids.append(attack_id)

        _append_if_new(list(self._attack_descriptions_en.keys()))
        _append_if_new(list(self._attack_descriptions_it.keys()))
        for mapping in self._config.get("default_mapping", []):
            _append_if_new(mapping.get("counter_attack_pool", []) or [])
        for theory in self._config.get("theories", []):
            _append_if_new(theory.get("default_counter_attacks", []) or [])
        for causal_type in self._config.get("causal_types", []):
            _append_if_new(causal_type.get("counter_attack_catalog", []) or [])
        return ids

    def _attack_description(
        self,
        attack_id: str,
        *,
        locale: str = "en",
        default: str = "",
    ) -> str:
        """Resolve attack description from taxonomy metadata with fallback."""
        if locale.lower().startswith("it"):
            return (
                self._attack_descriptions_it.get(attack_id)
                or self._attack_descriptions_en.get(attack_id)
                or default
            )
        return (
            self._attack_descriptions_en.get(attack_id)
            or self._attack_descriptions_it.get(attack_id)
            or default
        )

    def _resilient_model_order(self) -> list[str] | None:
        """Counter fallback chain from settings (selected model first)."""
        preferred_chain = settings.counter_model_fallback_order
        selected = settings.resolve_model_name(self.config.model_name)
        order = [selected] + [m for m in preferred_chain if m != selected]
        return order

    # ------------------------------------------------------------------
    # Attack selection logic (config-driven)
    # ------------------------------------------------------------------
    def _select_attacks(
        self, claim: str, routing_decision: RoutingDecision
    ) -> AttackSelection:
        """Select 2-3 counter attacks from config pools."""
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
            pool = theory_attacks or self._known_attack_ids()

        selected_ids = self._pick_attacks_with_llm(
            claim=claim,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            pool=pool,
        )
        if not selected_ids:
            fallback_count = min(3, len(pool))
            selected_ids = pool[:fallback_count]

        descriptions = {
            aid: self._attack_description(
                aid,
                locale="en",
                default=_DEFAULT_ATTACK_DESCRIPTION_EN,
            )
            for aid in selected_ids
        }
        return AttackSelection(
            pool=pool, attack_ids=selected_ids, descriptions=descriptions
        )

    def _pick_attacks_with_llm(
        self,
        claim: str,
        causal_type_id: str,
        theory_id: str,
        pool: List[str],
    ) -> List[str]:
        """Use LLM to pick 2-3 suitable attack ids from pool."""
        if not pool:
            return []

        if len(pool) == 1:
            return [pool[0]]

        min_attacks = 2
        max_attacks = min(3, len(pool))

        options_text = "\n".join(
            f"- {aid}: {self._attack_description(aid, locale='en', default=_DEFAULT_ATTACK_DESCRIPTION_EN)}"
            for aid in pool
        )
        prompt = render_prompt(
            "counter_reasoner.pick_attacks",
            claim=claim,
            causal_type_id=causal_type_id,
            theory_id=theory_id,
            min_attacks=min_attacks,
            max_attacks=max_attacks,
            options_text=options_text,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip()
            attack_ids = self._clean_attack_choices(
                raw=answer,
                pool=pool,
                min_count=min_attacks,
                max_count=max_attacks,
            )
            if attack_ids:
                return attack_ids
        except Exception as e:
            self._log(f"⚠️ LLM attack selection failed: {e}", "warning")

        return pool[:max_attacks]

    def _clean_attack_choices(
        self,
        raw: str,
        pool: List[str],
        min_count: int,
        max_count: int,
    ) -> List[str]:
        """Normalize LLM output to a list of valid attack IDs."""
        pool_set = set(pool)
        payload = raw.strip().replace("```json", "").replace("```", "").strip()
        candidates: List[str] = []

        if payload:
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    attack_ids = parsed.get("attack_ids", [])
                    if isinstance(attack_ids, list):
                        candidates.extend(str(a).strip() for a in attack_ids)
            except Exception:
                pass

        if not candidates:
            lowered = payload.lower()
            for attack_id in pool:
                if attack_id.lower() in lowered:
                    candidates.append(attack_id)

        cleaned: List[str] = []
        seen = set()
        for attack_id in candidates:
            normalized = attack_id.strip()
            if normalized not in pool_set or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
            if len(cleaned) >= max_count:
                break

        if len(cleaned) < min_count:
            for attack_id in pool:
                if attack_id in seen:
                    continue
                cleaned.append(attack_id)
                seen.add(attack_id)
                if len(cleaned) >= min_count:
                    break

        return cleaned[:max_count]

    def _expand_with_cross_references(self, statutes: List[dict]) -> List[dict]:
        """
        Add statutes explicitly referenced inside the text of already-retrieved articles.

        Forward-only: parses each article's text for patterns like
        "art. 624", "artt. 1223" and fetches the cited article by exact
        number from the KB via get_statute_by_article_tool.
        """
        try:
            import re
        except Exception:
            return statutes

        seen = {(s.get("articolo"), s.get("source")) for s in statutes}
        extra: List[dict] = []

        pattern = re.compile(r"art(?:t)?\.?\s*(\d{2,4})", re.IGNORECASE)

        for s in statutes:
            text = s.get("testo") or ""
            source = s.get("source", "codice_civile")

            refs = set(pattern.findall(text))

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
        reasoner_conclusion: str,
        enable_causality: bool = True,
        stream_callback: Optional[Callable[[dict], None]] = None,
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

        if not routing_decision:
            raise ValueError("routing_decision is required")
        if enable_causality and not routing_decision.causal_type_id:
            raise ValueError(
                "routing_decision with causal_type_id/theory_id is required when causality is enabled"
            )
        if not (reasoner_conclusion or "").strip():
            raise ValueError(
                "reasoner_conclusion is required for CounterReasoner (counter must oppose the Reasoner thesis)"
            )

        if enable_causality:
            # Select counter attacks from config pools
            attack_selection = self._select_attacks(claim, routing_decision)
            self._log(
                f"⚔️ Selected counter attacks: "
                f"{', '.join(attack_selection.attack_ids) if attack_selection.attack_ids else 'N/A'} "
                f"(pool size {len(attack_selection.pool)})"
            )
            if attack_selection.attack_ids:
                descs = [
                    f"{aid}: {attack_selection.descriptions.get(aid, 'N/A')}"
                    for aid in attack_selection.attack_ids
                ]
                self._log(f"📝 Attack descriptions: {' | '.join(descs)}")

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
            boosted_counter_statutes = self._retrieve_targeted_counter_boost(
                claim=claim,
                attack_ids=attack_selection.attack_ids,
                existing_statutes=pre_retrieved_statutes,
            )
        else:
            self._log(
                "🔬 Causality DISABLED — skipping attack selection and anchor norms"
            )
            attack_selection = AttackSelection(pool=[], attack_ids=[], descriptions={})
            anchor_statutes = []
            boosted_counter_statutes = []

        all_statutes = (
            pre_retrieved_statutes + boosted_counter_statutes + anchor_statutes
        )
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
            f"Art. {s.get('articolo')} ({self._source_short_label(s.get('source', ''))})"
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

            try:
                raw_output, iterative_chain, step_attack_ids = (
                    self._generate_counter_chain_iteratively(
                        claim=claim,
                        routing_decision=routing_decision,
                        attack_selection=attack_selection,
                        knowledge_base=knowledge_base,
                        allowed_statutes=allowed_statutes,
                        available_statutes=deduped_statutes,
                        allowed_precedents=allowed_precedents,
                        reasoner_conclusion=reasoner_conclusion,
                        stream_callback=stream_callback,
                    )
                )
            except Exception as gen_exc:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: counter planner/executor failed ({gen_exc})",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    raise
                continue

            # Build output
            output = CounterReasonerOutput(
                claim=claim,
                causal_type_id=routing_decision.causal_type_id,
                theory_id=routing_decision.theory_id,
                selected_attack_id=attack_selection.attack_id,
                selected_attack_ids=attack_selection.attack_ids,
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
                    "selected_attack_ids": attack_selection.attack_ids,
                    "selected_attack_by_step": [
                        {"step": idx + 1, "attack_id": attack_id}
                        for idx, attack_id in enumerate(step_attack_ids)
                    ],
                    "causal_type_id": routing_decision.causal_type_id,
                    "theory_id": routing_decision.theory_id,
                },
            )

            # Validate: ASPIC_IR must contain reasoning chain nodes (S1, S2, …)
            if self._has_valid_reasoning_chain(output.aspic_ir):
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

        chain_len = (
            len(output.aspic_ir.get("reasoning_chain", [])) if output.aspic_ir else 0
        )
        self._log(
            f"✅ Generated {len(output.counter_arguments)} counter-argument(s), "
            f"{chain_len} reasoning steps",
            "success",
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
        mentions = extract_article_mentions(text, require_code=True)
        articles: List[str] = [
            format_article_citation(m.article_id, m.source_hint) for m in mentions
        ]
        seen: set[str] = set()
        unique: List[str] = []
        for a in articles:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    @staticmethod
    def _article_result_to_dict(article) -> dict:
        """Convert ``ArticleResult`` into statute dict format used by agents."""
        return {
            "statute_id": article.statute_id,
            "articolo": article.articolo,
            "titolo": article.titolo,
            "testo": article.testo,
            "libro": article.libro,
            "source": article.source,
        }

    def _retrieve_targeted_counter_boost(
        self,
        claim: str,
        attack_ids: List[str],
        existing_statutes: List[dict],
        context_statutes_count: Optional[int] = None,
    ) -> List[dict]:
        """Run a second-pass retrieval guided by selected counter attacks."""
        if not settings.counter_second_pass_enabled:
            return []
        effective_context_size = (
            context_statutes_count
            if context_statutes_count is not None
            else len(existing_statutes)
        )
        if effective_context_size >= settings.counter_second_pass_min_against_statutes:
            return []

        hints: List[str] = []
        for attack_id in attack_ids:
            hint = self._attack_description(
                attack_id,
                locale="it",
                default=_DEFAULT_ATTACK_DESCRIPTION_IT,
            ).strip()
            if hint and hint not in hints:
                hints.append(hint)
        if not hints:
            hints.append("contestazione nesso causale e responsabilità")

        queries = [f"{claim}. Focus difensivo: {hint}." for hint in hints]
        queries = queries[: max(1, settings.counter_second_pass_max_queries)]

        self._log(
            "🔎 Counter second-pass retrieval attivato: "
            f"{len(queries)} query mirate da attack template",
            "info",
        )

        try:
            from services.legal_search import LegalSearchPipeline
        except Exception as exc:
            self._log(f"⚠️ Counter second-pass unavailable: {exc}", "warning")
            return []

        try:
            pipe = LegalSearchPipeline()
        except Exception as exc:
            self._log(f"⚠️ Counter second-pass init failed: {exc}", "warning")
            return []

        try:
            classification = pipe.classifier.classify(claim)
            filters = pipe.build_search_filters(
                classification,
                settings.search_use_top_n_libri,
            )

            boosted_by_id: Dict[str, dict] = {}
            existing_ids = {
                (s.get("statute_id") or "").strip()
                for s in existing_statutes
                if (s.get("statute_id") or "").strip()
            }

            for query_text in queries:
                embedding = pipe.embed_text(query_text)
                articles = pipe.vector_search(
                    embedding=embedding,
                    libri_filters=filters,
                    top_k=max(1, settings.counter_second_pass_top_k),
                    query_text=query_text,
                )
                articles = pipe.expand_with_cited_articles(articles)
                for article in articles:
                    statute_id = (article.statute_id or "").strip()
                    if not statute_id or statute_id in existing_ids:
                        continue
                    current = boosted_by_id.get(statute_id)
                    if current is None or float(article.score) > float(
                        current.get("_score", 0.0)
                    ):
                        payload = self._article_result_to_dict(article)
                        payload["_score"] = float(article.score)
                        boosted_by_id[statute_id] = payload

            boosted = sorted(
                boosted_by_id.values(),
                key=lambda x: float(x.get("_score", 0.0)),
                reverse=True,
            )
            boosted = boosted[: max(1, settings.counter_second_pass_max_additional)]
            for item in boosted:
                item.pop("_score", None)

            if not boosted:
                return []

            legal_context = self._extract_legal_context(claim)
            boosted = self.filter_irrelevant_statutes(claim, boosted)
            boosted = self.filter_applicable_statutes(claim, boosted, legal_context)

            self._log(
                f"✅ Counter second-pass: {len(boosted)} articoli aggiuntivi kept",
                "info",
            )
            return boosted

        except Exception as exc:
            self._log(f"⚠️ Counter second-pass retrieval failed: {exc}", "warning")
            return []

    def _generate_counter_chain_iteratively(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_selection: AttackSelection,
        knowledge_base: str,
        allowed_statutes: List[str],
        available_statutes: List[dict],
        allowed_precedents: List[str],
        reasoner_conclusion: str,
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> tuple[str, List[str], List[str]]:
        """Generate counter-reasoning chain with plan -> execute workflow."""
        max_steps = settings.chain_max_steps
        min_steps = settings.chain_min_steps
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )
        selected_attack_ids = attack_selection.attack_ids or [
            attack_selection.attack_id
        ]
        selected_attack_ids = [aid for aid in selected_attack_ids if aid]
        if not selected_attack_ids:
            selected_attack_ids = ["N/A"]
        attack_catalog = "\n".join(
            f"- {aid}: {self._attack_description(aid, locale='en', default=_DEFAULT_ATTACK_DESCRIPTION_EN)}"
            for aid in selected_attack_ids
        )

        plan = self._generate_counter_plan(
            claim=claim,
            routing_decision=routing_decision,
            selected_attack_ids=selected_attack_ids,
            attack_catalog=attack_catalog,
            reasoner_conclusion=reasoner_conclusion,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        self._log(f"🧭 Counter plan generated: {len(plan)} step(s)")

        steps: List[str] = []
        step_summaries: List[str] = []
        used_norms: List[str] = []
        step_attacks: List[str] = []
        allowed_statute_index = self._build_allowed_statute_index(available_statutes)

        for idx, plan_step in enumerate(plan, start=1):
            step_attack_id = plan_step.get("attack_id", selected_attack_ids[0])
            if step_attack_id not in selected_attack_ids:
                step_attack_id = selected_attack_ids[0]
            step_attack_desc = self._attack_description(
                step_attack_id,
                locale="en",
                default=_DEFAULT_ATTACK_DESCRIPTION_EN,
            )

            self._log(
                f"🔗 Generating planned counter-step {idx}/{len(plan)}: "
                f"{plan_step.get('goal', '')[:80]} | attack={step_attack_id}"
            )
            step_text = self._generate_counter_step_from_plan(
                claim=claim,
                routing_decision=routing_decision,
                attack_id=step_attack_id,
                attack_desc=step_attack_desc,
                reasoner_conclusion=reasoner_conclusion,
                knowledge_base=knowledge_base,
                statutes_list=statutes_list,
                precedents_list=precedents_list,
                plan=plan,
                plan_index=idx,
                plan_step=plan_step,
                previous_steps=steps,
                previous_summaries=step_summaries,
                used_norms=used_norms,
                allowed_statute_index=allowed_statute_index,
                stream_callback=stream_callback,
            )
            if not step_text:
                raise RuntimeError(
                    f"Planned counter step {idx} could not be generated with valid content"
                )
            steps.append(step_text)
            step_attacks.append(step_attack_id)
            step_summaries.append(self._compact_step_summary(step_text))
            new_norms = self._extract_cited_articles(step_text)
            for norm in new_norms:
                if norm not in used_norms:
                    used_norms.append(norm)

            prec_mentions = [
                p for p in allowed_precedents if p.lower() in step_text.lower()
            ]
            prec_info = f" | prec: {', '.join(prec_mentions)}" if prec_mentions else ""
            self._log(
                f"✅ Counter-step {idx}: {step_text[:80]}... "
                f"| attack: {step_attack_id} "
                f"| norms: {', '.join(new_norms) if new_norms else 'none'}{prec_info}"
            )

        if len(steps) < min_steps:
            raise RuntimeError(
                "Counter planner/executor produced fewer steps than chain_min_steps"
            )

        self._log(
            f"📊 Planned counter-chain complete: {len(steps)} steps, "
            f"{len(set(used_norms))} unique norms"
        )
        return (
            self._assemble_counter_raw_response(claim, steps, step_attacks),
            steps,
            step_attacks,
        )

    def _generate_counter_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        selected_attack_ids: List[str],
        attack_catalog: str,
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        min_steps: int,
        max_steps: int,
    ) -> List[Dict[str, str]]:
        """Generate and validate an execution plan for counter reasoning."""
        reasoner_block = f"\nReasoner conclusion to oppose:\n{reasoner_conclusion}\n"
        prompt = render_prompt(
            "counter_reasoner.generate_plan",
            claim=claim,
            reasoner_block=reasoner_block,
            routing_domain=routing_decision.domain,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            selected_attack_ids=", ".join(selected_attack_ids),
            attack_catalog=attack_catalog,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            knowledge_base=knowledge_base,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        last_error = "planner failed"
        for attempt in range(1, self._max_plan_retries + 1):
            try:
                resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
                raw = (resp.content or "").strip()
                plan = self._parse_counter_plan(
                    raw=raw,
                    min_steps=min_steps,
                    max_steps=max_steps,
                    allowed_attack_ids=selected_attack_ids,
                )
                if plan:
                    return plan
                last_error = "parsed empty plan"
            except Exception as e:
                last_error = str(e)
            self._log(
                f"⚠️ Counter planner attempt {attempt}/{self._max_plan_retries} failed: {last_error}",
                "warning",
            )
        raise RuntimeError(f"Counter planner failed: {last_error}")

    def _parse_counter_plan(
        self,
        raw: str,
        min_steps: int,
        max_steps: int,
        allowed_attack_ids: List[str],
    ) -> List[Dict[str, str]]:
        """Parse planner JSON and enforce plan quality constraints."""
        payload_text = raw.strip()
        if not payload_text.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", payload_text)
            if match:
                payload_text = match.group(0)
        data = json.loads(payload_text)
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list):
            raise ValueError("planner output missing 'steps' array")

        cleaned: List[Dict[str, str]] = []
        for idx, item in enumerate(steps_raw, start=1):
            if not isinstance(item, dict):
                continue
            goal = str(item.get("goal", "")).strip()
            focus = str(item.get("focus", "")).strip()
            expected_norm = str(item.get("expected_norm", "")).strip() or "N/A"
            citation_requirement = self._normalize_plan_citation_requirement(
                expected_norm=expected_norm,
                raw_value=item.get("citation_requirement"),
            )
            attack_id = str(item.get("attack_id", "")).strip()
            if not goal or not focus or not attack_id:
                continue
            if attack_id not in allowed_attack_ids:
                continue
            cleaned.append(
                {
                    "id": str(item.get("id", f"C{idx}")).strip() or f"C{idx}",
                    "goal": goal,
                    "focus": focus,
                    "expected_norm": expected_norm,
                    "citation_requirement": citation_requirement,
                    "attack_id": attack_id,
                }
            )

        if len(cleaned) < min_steps or len(cleaned) > max_steps:
            raise ValueError(
                f"invalid counter-plan length {len(cleaned)} (expected {min_steps}-{max_steps})"
            )
        if self._has_overlapping_plan_steps(cleaned):
            raise ValueError("counter planner produced overlapping/repetitive steps")

        min_distinct_attacks = min(2, len(allowed_attack_ids))
        distinct_attacks = {step.get("attack_id", "") for step in cleaned}
        if len(distinct_attacks) < min_distinct_attacks:
            raise ValueError(
                "counter planner did not distribute attacks across planned steps"
            )
        return cleaned

    @staticmethod
    def _normalize_plan_citation_requirement(
        *,
        expected_norm: str,
        raw_value: object,
    ) -> str:
        """Normalize planner citation policy with backward-compatible defaults."""
        value = str(raw_value or "").strip().lower()
        aliases = {
            "required": "required",
            "must": "required",
            "mandatory": "required",
            "optional": "optional",
            "if_possible": "optional",
            "when_possible": "optional",
            "none": "none",
            "no": "none",
        }
        normalized = aliases.get(value)
        if normalized:
            return normalized

        expected = str(expected_norm or "").strip().upper()
        if expected and expected not in {"N/A", "NA", "NONE", "-"}:
            return "required"
        return "optional"

    @staticmethod
    def _step_requires_citation(
        *, expected_norm: str, citation_requirement: str
    ) -> bool:
        """Decide citation requirement from planner metadata only."""
        requirement = str(citation_requirement or "").strip().lower()
        if requirement == "required":
            return True
        if requirement in {"optional", "none"}:
            return False
        expected = str(expected_norm or "").strip().upper()
        return bool(expected and expected not in {"N/A", "NA", "NONE", "-"})

    def _has_overlapping_plan_steps(self, plan_steps: List[Dict[str, str]]) -> bool:
        """Detect obvious overlap across planned goals/focuses."""
        normalized = []
        for step in plan_steps:
            text = f"{step.get('goal', '')} {step.get('focus', '')}".lower()
            text = re.sub(r"[^a-z0-9àèéìòù\s]", " ", text)
            words = {w for w in text.split() if len(w) > 3}
            normalized.append(words)

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                a = normalized[i]
                b = normalized[j]
                if not a or not b:
                    continue
                overlap = len(a & b) / len(a | b)
                if overlap >= 0.65:
                    return True
        return False

    def _generate_counter_step_from_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_id: str,
        attack_desc: str,
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        plan: List[Dict[str, str]],
        plan_index: int,
        plan_step: Dict[str, str],
        previous_steps: List[str],
        previous_summaries: List[str],
        used_norms: List[str],
        allowed_statute_index: Dict[str, set[str]],
        stream_callback: Optional[Callable[[dict], None]],
    ) -> str:
        """Execute one planned counter step with validation + retries."""
        base_prompt = self._build_counter_step_prompt_from_plan(
            claim=claim,
            routing_decision=routing_decision,
            attack_id=attack_id,
            attack_desc=attack_desc,
            reasoner_conclusion=reasoner_conclusion,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            plan=plan,
            plan_index=plan_index,
            plan_step=plan_step,
            previous_summaries=previous_summaries,
            used_norms=used_norms,
        )
        last_candidate = ""
        last_reason = "invalid output"
        for attempt in range(1, self._max_step_rewrites + 2):
            prompt = (
                base_prompt
                if attempt == 1
                else self._build_stance_rewrite_prompt(
                    original_prompt=base_prompt,
                    invalid_step=last_candidate,
                    invalid_reason=last_reason,
                )
            )
            try:
                resp = self._resilient_llm_invoke(
                    [HumanMessage(content=prompt)],
                    stream_callback=(
                        (
                            lambda token: self._emit_stream_token(
                                stream_callback,
                                phase="counter",
                                token=token,
                                step=plan_index,
                            )
                        )
                        if stream_callback and attempt == 1
                        else None
                    ),
                )
                candidate = self._parse_step_text((resp.content or "").strip())
            except Exception as exc:
                last_reason = f"generation error: {exc}"
                self._log(
                    f"⚠️ Counter-step {plan_index} generation failed (attempt {attempt}): {exc}",
                    "warning",
                )
                continue

            last_candidate = candidate
            ok, reason = self._validate_counter_step_candidate(
                candidate_step=candidate,
                previous_steps=previous_steps,
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                expected_norm=plan_step.get("expected_norm", "N/A"),
                citation_requirement=plan_step.get("citation_requirement", "optional"),
                allowed_statute_index=allowed_statute_index,
                attack_id=plan_step.get("attack_id", attack_id),
                attack_desc=attack_desc,
                plan_focus=plan_step.get("focus", ""),
            )
            if ok:
                return candidate
            last_reason = reason
            if stream_callback:
                try:
                    stream_callback(
                        {
                            "phase": "counter",
                            "action": "reset_step",
                            "step": plan_index,
                        }
                    )
                except Exception:
                    pass
            self._log(
                f"⚠️ Counter-step {plan_index} rejected ({reason}) "
                f"[attempt {attempt}/{self._max_step_rewrites + 1}]",
                "warning",
            )
        return ""

    def _build_counter_step_prompt_from_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_id: str,
        attack_desc: str,
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        plan: List[Dict[str, str]],
        plan_index: int,
        plan_step: Dict[str, str],
        previous_summaries: List[str],
        used_norms: List[str],
    ) -> str:
        """Create prompt for one planned counter step."""
        plan_lines = "\n".join(
            f"{idx}. {step.get('goal', '')} | focus: {step.get('focus', '')} | attack: {step.get('attack_id', '')}"
            for idx, step in enumerate(plan, start=1)
        )
        summary_lines = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(previous_summaries, start=1)
            )
            if previous_summaries
            else "- none"
        )
        used_norms_text = ", ".join(used_norms) if used_norms else "none"
        reasoner_block = f"\nCONCLUSION TO OPPOSE:\n{reasoner_conclusion}\n"
        return render_prompt(
            "counter_reasoner.step_prompt",
            claim=claim,
            reasoner_block=reasoner_block,
            routing_domain=routing_decision.domain,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            attack_id=attack_id,
            attack_desc=attack_desc,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            plan_lines=plan_lines,
            plan_index=plan_index,
            plan_goal=plan_step.get("goal", ""),
            plan_focus=plan_step.get("focus", ""),
            plan_expected_norm=plan_step.get("expected_norm", "N/A"),
            plan_citation_requirement=plan_step.get("citation_requirement", "optional"),
            plan_attack_id=plan_step.get("attack_id", attack_id),
            summary_lines=summary_lines,
            used_norms_text=used_norms_text,
        )

    def _validate_counter_step_candidate(
        self,
        candidate_step: str,
        previous_steps: List[str],
        claim: str,
        reasoner_conclusion: str,
        expected_norm: str,
        citation_requirement: str,
        allowed_statute_index: Dict[str, set[str]],
        attack_id: str,
        attack_desc: str,
        plan_focus: str,
    ) -> tuple[bool, str]:
        """Validation checks for one counter step candidate."""
        text = (candidate_step or "").strip()
        if not text or text.upper() == "DONE":
            return False, "empty step"
        if self._is_garbage_text(text):
            return False, "garbage output"
        if not self._is_counter_step_consistent(text):
            return False, "reasoning inconsistency"
        facts_ok, facts_reason = self._is_counter_step_fact_consistent_with_claim(
            claim=claim,
            candidate_step=text,
        )
        if not facts_ok:
            return False, facts_reason
        opposes_ok, opposes_reason = self._is_counter_step_opposed_to_reasoner(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            candidate_step=text,
        )
        if not opposes_ok:
            return False, opposes_reason
        mention_matches = extract_article_mentions(text, require_code=False)
        if (
            self._step_requires_citation(
                expected_norm=expected_norm,
                citation_requirement=citation_requirement,
            )
            and not mention_matches
        ):
            return False, "missing statutory citation"
        if mention_matches:
            grounded, grounded_reason = self._has_grounded_citation(
                mentions=mention_matches,
                allowed_statute_index=allowed_statute_index,
            )
            if not grounded:
                return False, grounded_reason
            expected_ok, expected_reason = self._matches_expected_norm(
                mentions=mention_matches,
                expected_norm=expected_norm,
            )
            if not expected_ok:
                return False, expected_reason
        compatible, reason = self._is_counter_step_compatible_with_history(
            candidate_step=text,
            previous_steps=previous_steps,
            claim=claim,
        )
        if not compatible:
            return False, reason or "history incompatibility"
        if previous_steps and self._is_repetitive_step(text, previous_steps):
            return False, "lexical repetition"
        if previous_steps and self._is_semantically_redundant_step(
            candidate_step=text,
            previous_steps=previous_steps,
            claim=claim,
            role="counter",
        ):
            return False, "semantic repetition"
        aligned, alignment_reason = self._is_step_aligned_with_attack(
            claim=claim,
            candidate_step=text,
            attack_id=attack_id,
            attack_desc=attack_desc,
            plan_focus=plan_focus,
        )
        if not aligned:
            return False, alignment_reason
        return True, ""

    def _is_counter_step_fact_consistent_with_claim(
        self,
        *,
        claim: str,
        candidate_step: str,
    ) -> tuple[bool, str]:
        """Reject counter-steps that contradict explicit facts stated in the claim.

        The counter can attack legal qualification and inference, but must not
        negate factual premises expressly given in the claim text.
        """
        return self._is_step_fact_consistent_with_claim(
            claim=claim,
            candidate_step=candidate_step,
            actor_label="CounterReasoner",
        )

    def _is_counter_step_opposed_to_reasoner(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        candidate_step: str,
    ) -> tuple[bool, str]:
        """Reject candidate counter-steps that materially agree with the Reasoner conclusion.

        This is a semantic check relative to the opposing thesis, not a polarity
        check on the claim itself.
        """
        reasoner_text = (reasoner_conclusion or "").strip()
        if not reasoner_text:
            return False, "missing reasoner conclusion context"

        step_text = (candidate_step or "").strip()
        if not step_text:
            return False, "empty step"

        cache_key = (reasoner_text, step_text)
        cached = self._reasoner_opposition_check_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "counter_reasoner.step_opposition_check",
            claim=claim,
            reasoner_conclusion=reasoner_text,
            candidate_step=step_text,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip().upper()
            if "AGREE" in answer:
                result = (False, "agrees with reasoner conclusion")
            else:
                # UNCLEAR is accepted here to avoid over-rejecting premise-level attacks.
                result = (True, "")
        except Exception as exc:
            self._log(
                f"⚠️ Counter opposition check failed (fallback keep): {exc}",
                "warning",
            )
            result = (True, "")

        self._reasoner_opposition_check_cache[cache_key] = result
        return result

    @staticmethod
    def _normalize_source_for_match(source_raw: str) -> str:
        """Normalize source labels to internal statute-source keys."""
        source = (source_raw or "").strip().lower()
        if not source:
            return ""
        if source in {"codice_amministrativo", "amministrativo"}:
            return "codice_amministrativo"
        if source in {"codice_civile", "civile"}:
            return "codice_civile"
        if source in {"codice_penale", "penale"}:
            return "codice_penale"
        if "241" in source or "l. 241/1990" in source or "amm" in source:
            return "codice_amministrativo"
        if "c.c" in source or ("cod" in source and "civ" in source):
            return "codice_civile"
        if "c.p" in source or ("cod" in source and "pen" in source):
            return "codice_penale"
        return source

    def _build_allowed_statute_index(self, statutes: List[dict]) -> Dict[str, set[str]]:
        """
        Build article->source index for grounding checks.

        Example:
            {"21-octies": {"codice_amministrativo"}}
        """
        index: Dict[str, set[str]] = {}
        for statute in statutes:
            article = normalize_article_id(str(statute.get("articolo", "")))
            if not article:
                continue
            source = self._normalize_source_for_match(str(statute.get("source", "")))
            if article not in index:
                index[article] = set()
            if source:
                index[article].add(source)
        return index

    def _has_grounded_citation(
        self,
        mentions: List,
        allowed_statute_index: Dict[str, set[str]],
    ) -> tuple[bool, str]:
        """
        Ensure cited articles are grounded in the allowed counter statute set.
        """
        if not mentions:
            return False, "missing statutory citation"
        if not allowed_statute_index:
            return True, ""

        grounded = 0
        ungrounded_details: List[str] = []
        seen_keys = set()

        for mention in mentions:
            article_id = normalize_article_id(getattr(mention, "article_id", ""))
            if not article_id:
                continue
            mention_source = self._normalize_source_for_match(
                getattr(mention, "source_hint", "")
                or infer_source_hint(getattr(mention, "raw_code", ""))
            )
            key = (article_id, mention_source)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            allowed_sources = allowed_statute_index.get(article_id)
            if not allowed_sources:
                ungrounded_details.append(f"Art. {article_id} not in allowed set")
                continue
            if mention_source:
                if mention_source in allowed_sources:
                    grounded += 1
                else:
                    ungrounded_details.append(
                        f"Art. {article_id} source mismatch ({mention_source})"
                    )
                continue

            # No source provided in text: allow only if article is unambiguous.
            if len(allowed_sources) == 1:
                grounded += 1
            else:
                ungrounded_details.append(
                    f"Art. {article_id} ambiguous source (missing code)"
                )

        if grounded == 0:
            return False, "citation not grounded in allowed statutes"
        if ungrounded_details:
            return (
                False,
                "contains ungrounded citation(s): " + "; ".join(ungrounded_details[:2]),
            )
        return True, ""

    def _matches_expected_norm(
        self,
        mentions: List,
        expected_norm: str,
    ) -> tuple[bool, str]:
        """Check whether candidate cites the expected norm planned for this step."""
        expected = (expected_norm or "").strip()
        if not expected or expected.upper() in {"N/A", "NA", "NONE", "-"}:
            return True, ""

        expected_mentions = extract_article_mentions(expected, require_code=False)
        expected_ids = {
            normalize_article_id(m.article_id)
            for m in expected_mentions
            if m.article_id
        }
        if not expected_ids:
            # Planner can emit non-parseable labels; don't hard-fail in that case.
            return True, ""

        cited_ids = {
            normalize_article_id(getattr(m, "article_id", ""))
            for m in mentions
            if getattr(m, "article_id", "")
        }
        if expected_ids & cited_ids:
            return True, ""
        return (
            False,
            "step does not cite expected norm "
            f"({expected_norm}); cited={', '.join(sorted(cited_ids)) or 'none'}",
        )

    def _is_step_aligned_with_attack(
        self,
        claim: str,
        candidate_step: str,
        attack_id: str,
        attack_desc: str,
        plan_focus: str,
    ) -> tuple[bool, str]:
        """
        Verify that the generated step actually executes the assigned attack.
        """
        prompt = render_prompt(
            "counter_reasoner.attack_alignment",
            claim=claim,
            attack_id=attack_id,
            attack_desc=attack_desc,
            plan_focus=plan_focus,
            candidate_step=candidate_step,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip().upper()
            if "MISALIGNED" in answer:
                return False, "attack-plan misalignment"
            if "ALIGNED" in answer:
                return True, ""
        except Exception as exc:
            self._log(
                f"⚠️ Attack alignment check failed, fallback heuristic in use: {exc}",
                "warning",
            )

        # Conservative lexical fallback when LLM output is unavailable/ambiguous.
        focus_terms = {
            t
            for t in re.findall(
                r"[a-zàèéìòù]{5,}", f"{attack_desc} {plan_focus}".lower()
            )
            if t not in {"normativa", "giuridica", "procedimento", "provvedimento"}
        }
        if not focus_terms:
            return True, ""
        step_text = (candidate_step or "").lower()
        if any(term in step_text for term in focus_terms):
            return True, ""
        return False, "attack-plan misalignment (focus not reflected in step)"

    def _is_semantically_redundant_step(
        self,
        candidate_step: str,
        previous_steps: List[str],
        claim: str,
        role: str,
    ) -> bool:
        """LLM-based semantic redundancy check (NEW vs REPEAT)."""
        if not previous_steps:
            return False
        context_prev = "\n".join(
            f"{idx}. {step}" for idx, step in enumerate(previous_steps[-3:], start=1)
        )
        prompt = render_prompt(
            "counter_reasoner.semantic_redundancy",
            claim=claim,
            role=role,
            context_prev=context_prev,
            candidate_step=candidate_step,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip().upper()
            return "REPEAT" in answer
        except Exception:
            return False

    @staticmethod
    def _compact_step_summary(step_text: str) -> str:
        """Compact summary used as execution memory for following steps."""
        first_sentence = re.split(r"(?<=[.!?])\s+", step_text.strip())[0]
        first_sentence = re.sub(r"\s+", " ", first_sentence).strip()
        return first_sentence[:220]

    @staticmethod
    def _emit_stream_token(
        stream_callback: Optional[Callable[[dict], None]],
        *,
        phase: str,
        token: str,
        step: Optional[int] = None,
    ) -> None:
        """Emit one token chunk to external streaming callback."""
        if not stream_callback or not token:
            return
        payload: dict[str, str | int] = {"phase": phase, "token": token}
        if step is not None:
            payload["step"] = step
        try:
            stream_callback(payload)
        except Exception:
            # Streaming callback errors must never break counter-generation.
            pass

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

        # FIX: Strip metadata leakage (prompt context echoed by LLM)
        step_text = re.sub(
            r"\s*(?:Norms already used|Current step|New norm):.*$",
            "",
            step_text,
            flags=re.DOTALL,
        ).strip()

        return step_text

    def _build_stance_rewrite_prompt(
        self, original_prompt: str, invalid_step: str, invalid_reason: str = ""
    ) -> str:
        """Ask the model to rewrite a step that violates counter-step consistency rules."""
        reason_text = invalid_reason or "it is not a coherent counter-step."
        return render_prompt(
            "counter_reasoner.stance_rewrite",
            original_prompt=original_prompt,
            invalid_reason=reason_text,
            invalid_step=invalid_step,
        )

    def _is_counter_step_consistent(self, step_text: str) -> bool:
        """
        Lightweight guardrail against self-contradictory legal reasoning text.

        This intentionally does NOT enforce claim-polarity framing. Opposition to the
        Reasoner conclusion (when available) is checked separately.
        """
        return self._is_step_self_consistent(step_text)

    @staticmethod
    def _extract_step_signals(text: str) -> Dict[str, int]:
        """
        Extract coarse factual/legal polarity signals from a step.

        Signals are used to reject steps that contradict already accepted
        counter-steps or invert explicit claim facts.
        """
        t = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not t:
            return {
                "complexity": 0,
                "contestation": 0,
                "communication_need": 0,
                "legitimacy_effect": 0,
            }

        def has(patterns: List[str]) -> bool:
            return any(re.search(p, t) for p in patterns)

        complexity = 0
        if has([r"\bcompless", r"\barticolat", r"\bnon\s+lineare"]):
            complexity = 1
        elif has([r"\bsemplic", r"\blineare", r"\bchiar[ao]\b"]):
            complexity = -1

        contestation = 0
        if has([r"\bcontestat", r"\bcontrovers", r"\bdisputat"]):
            if not has([r"\bnon\s+contestat", r"\bincontestat"]):
                contestation = 1
        elif has([r"\bnon\s+contestat", r"\bincontestat", r"\bpacific"]):
            contestation = -1

        communication_need = 0
        if has(
            [
                r"\bcomunicazione\b.*\b(?:necessaria|obbligatoria|imposta)\b",
                r"\bart\.\s*7\b.*\bimpone\b",
            ]
        ):
            communication_need = 1
        if has(
            [
                r"\bcomunicazione\b.*\bnon\s+(?:necessaria|obbligatoria)\b",
                r"\bmancata\s+comunicazione\b.*\bnon\s+incide\b",
            ]
        ):
            communication_need = -1

        legitimacy_effect = 0
        if has(
            [
                r"\bincide\b.*\blegittimit",
                r"\bdetermina\b.*\billegittimit",
                r"\brende\b.*\billegittim",
                r"\bprovvedimento\b.*\billegittim",
            ]
        ) and not has(
            [
                r"\bnon\s+incide\b.*\blegittimit",
                r"\bnon\s+determina\b.*\billegittimit",
                r"\bnon\s+rende\b.*\billegittim",
                r"\bnon\s+(?:e|è)?\s*annullabil",
            ]
        ):
            legitimacy_effect = 1
        elif has(
            [
                r"\bnon\s+incide\b.*\blegittimit",
                r"\bnon\s+determina\b.*\billegittimit",
                r"\bnon\s+rende\b.*\billegittim",
                r"\bnon\s+(?:e|è)?\s*annullabil",
            ]
        ):
            legitimacy_effect = -1

        return {
            "complexity": complexity,
            "contestation": contestation,
            "communication_need": communication_need,
            "legitimacy_effect": legitimacy_effect,
        }

    def _is_counter_step_compatible_with_history(
        self,
        candidate_step: str,
        previous_steps: List[str],
        claim: str,
    ) -> tuple[bool, str]:
        """
        Check whether candidate step is logically compatible with history and claim facts.
        """
        candidate = self._extract_step_signals(candidate_step)
        claim_signals = self._extract_step_signals(claim)

        # Hard lock on explicit claim facts for complexity/contestation.
        for key in ("complexity", "contestation"):
            c_sig = claim_signals.get(key, 0)
            s_sig = candidate.get(key, 0)
            if c_sig and s_sig and c_sig != s_sig:
                return False, f"candidate flips claim fact '{key}'"

        if not previous_steps:
            return True, ""

        history_sum: Dict[str, int] = {
            "complexity": 0,
            "contestation": 0,
            "communication_need": 0,
            "legitimacy_effect": 0,
        }
        for step in previous_steps:
            sig = self._extract_step_signals(step)
            for key, value in sig.items():
                history_sum[key] += value

        # Hard-consistency on factual premises only.
        for key in ("complexity", "contestation"):
            h_sig = history_sum.get(key, 0)
            s_sig = candidate.get(key, 0)
            if h_sig and s_sig and (h_sig * s_sig < 0):
                return False, f"candidate contradicts chain on '{key}'"

        # Soft dimensions can legitimately vary across different counter-angles.
        for key in ("communication_need", "legitimacy_effect"):
            h_sig = history_sum.get(key, 0)
            s_sig = candidate.get(key, 0)
            if h_sig and s_sig and (h_sig * s_sig < 0):
                self._log(
                    f"ℹ️ Counter-step soft divergence on '{key}' accepted "
                    "(different attack angle).",
                    "info",
                )

        return True, ""

    def _derive_counter_conclusion_ground(self, steps: List[str]) -> str:
        """Pick a final counter-rationale from the last coherent step."""
        for step in reversed(steps):
            if not self._is_counter_step_consistent(step):
                continue
            first_sentence = re.split(r"(?<=[.!?])\s+", step.strip())[0]
            first_sentence = re.sub(
                r"^(?:pertanto|quindi|dunque|in\s+conclusione)\s*,?\s*",
                "",
                first_sentence,
                flags=re.IGNORECASE,
            ).strip()
            first_sentence = first_sentence.rstrip(" .")
            if first_sentence:
                return first_sentence
        return (
            "le norme richiamate e i fatti allegati non giustificano "
            "in modo univoco la tesi principale"
        )

    def _assemble_counter_raw_response(
        self,
        claim: str,
        steps: List[str],
        step_attack_ids: List[str],
    ) -> str:
        """Assemble counter-argument raw response from iterative steps."""
        chain_section = "**Catena di ragionamento**:\n"
        for i, step in enumerate(steps, 1):
            chain_section += f"{i}. {step}\n"

        premise_steps = steps[:-1] if len(steps) > 1 else steps

        premise_text = " ".join(premise_steps)
        norms = self._extract_cited_articles(" ".join(steps))
        norms_text = "\n".join(f"- {n}" for n in norms) if norms else "N/D"
        conclusion_ground = self._derive_counter_conclusion_ground(steps)

        unique_attacks = list(dict.fromkeys(a for a in step_attack_ids if a))
        attack_desc_list = [
            self._attack_description(
                attack_id,
                locale="it",
                default=_DEFAULT_ATTACK_DESCRIPTION_IT,
            )
            for attack_id in unique_attacks
        ]
        if not attack_desc_list:
            attack_desc_list = [_DEFAULT_ATTACK_DESCRIPTION_IT]
        attack_desc_it = "; ".join(attack_desc_list[:3])

        conclusion_text = (
            "Pertanto, la tesi giuridica principale deve essere contestata o "
            f"ridimensionata poiché {conclusion_ground}."
        )

        raw = (
            f"**Premessa Alternativa**: {premise_text}\n\n"
            f"**Norma**:\n{norms_text}\n\n"
            f"**Nesso Causale Alternativo**: L'analisi giuridica dimostra che "
            f"{attack_desc_it}. La catena argomentativa evidenzia come "
            f"le norme applicabili al caso consentano una ricostruzione alternativa "
            f"o limitativa rispetto alla tesi principale.\n\n"
            f"**Conclusione Contraria**: {conclusion_text}\n\n"
            f"{chain_section}"
        )
        return raw

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
    attack_ids: List[str]
    descriptions: Dict[str, str]

    @property
    def attack_id(self) -> str:
        """Primary attack id (first selected), kept for compatibility."""
        return self.attack_ids[0] if self.attack_ids else ""

    @property
    def description(self) -> str:
        """Primary attack description (first selected), kept for compatibility."""
        return self.descriptions.get(self.attack_id, "")
