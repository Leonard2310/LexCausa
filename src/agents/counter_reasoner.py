"""
LexCausa Counter-Reasoner Agent.

Generates counter-arguments on the same claim, targeting the
Reasoner's conclusion (required input).
Receives the claim + causal_type_id/theory_id from the Router and the
pre-retrieved relevant sources.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
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
from .tools.neo4j_tools import get_legal_search_pipeline, get_statute_by_article_tool
from .tools.prompt_registry import get_prompt, render_prompt
from .tools.taxonomy_tools import get_causality_theory_tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq  # noqa: E402
from services.pipeline_control import PipelineCancelled  # noqa: E402


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


_DEFAULT_ATTACK_DESCRIPTION_EN = "Counter-argument to weaken the primary legal thesis."
_DEFAULT_ATTACK_DESCRIPTION_IT = (
    "le norme citate indeboliscono la tesi giuridica primaria"
)


class CounterReasoner(BaseAgent):
    """
    Legal Counter-Reasoner Agent.

    Generates counter-arguments against the primary legal thesis,
    using the Reasoner's conclusion as mandatory target.
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
        self._new_facts_check_cache: Dict[tuple[str, str], tuple[bool, str]] = {}
        self._target_map_cache: Dict[tuple[str, str], dict] = {}

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

    @staticmethod
    def _claim_fact_anchors(claim: str, max_items: int = 8) -> List[str]:
        """Extract compact factual anchors from claim text for prompt grounding."""
        text = re.sub(r"\s+", " ", (claim or "").strip())
        if not text:
            return []
        chunks = re.split(r"(?<=[.;])\s+|,\s+(?=[A-Za-zÀ-ÿ])", text)
        anchors: List[str] = []
        seen = set()
        for chunk in chunks:
            c = chunk.strip(" .;:-")
            if len(c) < 18:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            anchors.append(c)
            if len(anchors) >= max_items:
                break
        return anchors

    def _claim_fact_anchors_text(self, claim: str, max_items: int = 8) -> str:
        anchors = self._claim_fact_anchors(claim, max_items=max_items)
        if not anchors:
            return "- none"
        return "\n".join(f"- {a}" for a in anchors)

    @staticmethod
    def _is_attack_compatible_with_claim(attack_id: str, claim: str) -> bool:
        """Basic sanity check for attack ids."""
        return bool((attack_id or "").strip())

    def _attack_definition_meta(self, attack_id: str) -> Dict[str, object]:
        """Return attack metadata from taxonomy definitions."""
        defs = self._config.get("counter_attack_definitions", {})
        meta = defs.get(attack_id, {})
        return meta if isinstance(meta, dict) else {}

    def _extract_counter_target_map(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
    ) -> Dict[str, List[str]]:
        """Extract target map for planner scope control."""
        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[:1500]
        reasoner_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[:1000]
        if not claim_text:
            return {
                "allowed_targets": [],
                "forbidden_assumptions": [],
                "priority_targets": [],
            }
        cache_key = (claim_text, reasoner_text)
        cached = self._target_map_cache.get(cache_key)
        if cached is not None:
            return cached
        prompt = render_prompt(
            "counter_reasoner.target_map",
            claim=claim_text,
            reasoner_conclusion=reasoner_text,
        )
        fallback = {
            "allowed_targets": [],
            "forbidden_assumptions": [],
            "priority_targets": [],
        }
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            raw = (resp.content or "").strip()
            if not raw.startswith("{"):
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    raw = m.group(0)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = {}
            for k in ("allowed_targets", "forbidden_assumptions", "priority_targets"):
                vals = parsed.get(k, [])
                if not isinstance(vals, list):
                    vals = []
                fallback[k] = [
                    str(v).strip() for v in vals if isinstance(v, str) and str(v).strip()
                ][:10]
        except Exception as exc:
            self._log(f"⚠️ Counter target-map extraction failed: {exc}", "warning")
        self._target_map_cache[cache_key] = fallback
        return fallback

    @staticmethod
    def _target_map_text(target_map: Dict[str, List[str]]) -> str:
        """Serialize target map into concise planner text."""
        if not isinstance(target_map, dict):
            return "- none"
        allowed = target_map.get("allowed_targets", []) or []
        forbidden = target_map.get("forbidden_assumptions", []) or []
        priority = target_map.get("priority_targets", []) or []
        parts: List[str] = []
        if allowed:
            parts.append("Allowed targets:\n" + "\n".join(f"- {x}" for x in allowed))
        if priority:
            parts.append("Priority targets:\n" + "\n".join(f"- {x}" for x in priority))
        if forbidden:
            parts.append(
                "Forbidden assumptions:\n" + "\n".join(f"- {x}" for x in forbidden)
            )
        return "\n\n".join(parts) if parts else "- none"

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
        self,
        claim: str,
        routing_decision: RoutingDecision,
        reasoner_conclusion: str,
    ) -> AttackSelection:
        """Select 2-3 counter attacks from config pools."""
        raw_causal_pool: List[str] = config_loader.counter_attack_pool_for(
            routing_decision.causal_type_id, self._config
        )
        raw_theory_attacks: List[str] = config_loader.theory_counter_attacks(
            routing_decision.theory_id, self._config
        )

        def _compatible(aid: str) -> bool:
            return self._is_attack_compatible_with_claim(aid, claim)
        causal_pool = [
            aid
            for aid in raw_causal_pool
            if _compatible(aid)
        ]
        theory_attacks = [
            aid
            for aid in raw_theory_attacks
            if _compatible(aid)
        ]
        pool: List[str] = list(causal_pool)

        if theory_attacks:
            intersection = [a for a in pool if a in theory_attacks]
            # Keep theory-consistent attacks first, but do not collapse the pool:
            # over-pruning to a tiny intersection makes the planner brittle.
            if intersection:
                ordered_pool: List[str] = []
                for aid in intersection + pool + theory_attacks:
                    if aid and aid not in ordered_pool:
                        ordered_pool.append(aid)
                pool = ordered_pool

        if not pool:
            # Fallback to theory attacks or all known attacks
            known = [
                aid
                for aid in self._known_attack_ids()
                if _compatible(aid)
            ]
            pool = theory_attacks or known

        selected_ids = self._pick_attacks_with_llm(
            claim=claim,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            pool=pool,
        )
        if not selected_ids:
            fallback_count = min(3, len(pool))
            selected_ids = pool[:fallback_count]

        # Prefer 3 active attacks when available to avoid early collapse.
        if len(pool) >= 3:
            min_target = 3
        elif len(pool) >= 2:
            min_target = 2
        else:
            min_target = 1

        feasible_selected = self._filter_feasible_attacks(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            candidate_attack_ids=selected_ids,
        )
        if len(feasible_selected) < min_target:
            # Second chance: enrich with feasibility over the broader pool.
            feasible_from_pool = self._filter_feasible_attacks(
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                candidate_attack_ids=pool[: min(10, len(pool))],
            )
            merged: List[str] = list(feasible_selected)
            for aid in feasible_from_pool:
                if aid not in merged:
                    merged.append(aid)
            feasible_selected = merged

        if feasible_selected:
            selected_ids = feasible_selected[: min(3, len(feasible_selected))]
            if len(selected_ids) < min_target:
                for aid in pool:
                    if aid in selected_ids:
                        continue
                    selected_ids.append(aid)
                    if len(selected_ids) >= min_target:
                        break
        else:
            selected_ids = []
            # Taxonomy fallback ladder before open-attack mode:
            # 1) remaining causal attacks
            # 2) remaining theory attacks
            fallback_taxonomy_ids: List[str] = []
            for aid in list(causal_pool) + list(theory_attacks):
                if aid and aid not in pool and aid not in fallback_taxonomy_ids:
                    fallback_taxonomy_ids.append(aid)
            if fallback_taxonomy_ids:
                feasible_fallback = self._filter_feasible_attacks(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    candidate_attack_ids=fallback_taxonomy_ids[:12],
                )
                if feasible_fallback:
                    selected_ids = feasible_fallback[: min(3, len(feasible_fallback))]
                    self._log(
                        "Warning: primary taxonomy attacks infeasible; using fallback taxonomy attacks",
                        "warning",
                    )

        descriptions = {
            aid: self._attack_description(
                aid,
                locale="en",
                default=_DEFAULT_ATTACK_DESCRIPTION_EN,
            )
            for aid in selected_ids
        }
        return AttackSelection(
            pool=pool,
            attack_ids=selected_ids,
            descriptions=descriptions,
            causal_pool=causal_pool,
            theory_pool=theory_attacks,
        )

    def _select_open_attacks(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        min_attacks: int = 2,
        max_attacks: int = 3,
    ) -> AttackSelection:
        """
        Select attack strategies without taxonomy IDs (open mode).
        """
        prompt = render_prompt(
            "counter_reasoner.open_attacks",
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            min_attacks=min_attacks,
            max_attacks=max_attacks,
        )
        attacks_raw: List[dict] = []
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            payload = (
                (resp.content or "")
                .strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and isinstance(parsed.get("attacks"), list):
                attacks_raw = [a for a in parsed["attacks"] if isinstance(a, dict)]
        except Exception as exc:
            self._log(f"⚠️ Open attack generation failed: {exc}", "warning")

        normalized: List[tuple[str, str]] = []
        seen_ids = set()
        for idx, item in enumerate(attacks_raw, start=1):
            raw_id = str(item.get("id", "")).strip().lower()
            raw_desc = str(item.get("description", "")).strip()
            attack_id = re.sub(r"[^a-z0-9_]+", "_", raw_id).strip("_")
            if not attack_id:
                attack_id = f"open_attack_{idx}"
            if len(attack_id) > 40:
                attack_id = attack_id[:40].rstrip("_")
            if not raw_desc:
                continue
            if attack_id in seen_ids:
                continue
            seen_ids.add(attack_id)
            normalized.append((attack_id, raw_desc))
            if len(normalized) >= max_attacks:
                break

        if len(normalized) < min_attacks:
            fallback = [
                (
                    "open_evidence_weight",
                    "Contestare sufficienza e univocità probatoria del nesso causale senza negare i fatti espliciti.",
                ),
                (
                    "open_subjective_element",
                    "Ridimensionare il rimprovero soggettivo e il grado di colpa alla luce dell'urgenza familiare dichiarata.",
                ),
                (
                    "open_sanction_balancing",
                    "Controbilanciare aggravanti e attenuanti per limitare la portata della conclusione punitiva.",
                ),
            ]
            normalized = fallback[:max_attacks]

        # Strict feasibility pass in open mode: keep ONLY FEASIBLE attacks.
        feasible = self._filter_feasible_open_attacks(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            attacks=normalized,
        )
        normalized = feasible[:max_attacks]

        pool = [aid for aid, _ in normalized]
        descriptions = {aid: desc for aid, desc in normalized}
        return AttackSelection(pool=pool, attack_ids=pool, descriptions=descriptions)

    def _filter_feasible_open_attacks(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        attacks: List[tuple[str, str]],
    ) -> List[tuple[str, str]]:
        """Normalize open attacks list (feasibility filter removed)."""
        if not attacks:
            return []
        _ = (claim, reasoner_conclusion)
        return attacks

    def _filter_feasible_attacks(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        candidate_attack_ids: List[str],
    ) -> List[str]:
        """Return candidate attacks in original order (feasibility filter removed)."""
        _ = (claim, reasoner_conclusion)
        return [a for a in dict.fromkeys(candidate_attack_ids) if a]

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
        Get taxonomy anchor statutes for the given causal types.

        Alignment with Reasoner:
        - taxonomy claim-relevance filter via get_causality_theory_tool(claim=...)
        - applicability filter on converted statutes (soft core / hard accessory)
        """
        statutes: List[dict] = []
        seen_refs = set()
        unique_cts = list(dict.fromkeys(ct for ct in causal_types if ct))

        for ct in unique_cts:
            try:
                theory = get_causality_theory_tool.invoke(
                    {"causality_type": ct, "claim": claim}
                )
            except Exception as e:
                self._log(f"⚠️ Failed to load theory for {ct}: {e}", "warning")
                continue

            core_rel = theory.get("norme_core_rilevanti", []) or []
            acc_rel = theory.get("norme_accessorie_rilevanti", []) or []
            core_full = theory.get("norme_core", []) or []
            acc_full = theory.get("norme_accessorie", []) or []

            if not core_rel and core_full:
                core_rel = core_full
            if not acc_rel and acc_full:
                acc_rel = acc_full

            taxonomy_norms = core_rel + acc_rel
            kept_refs = [
                n.get("ref") or n.get("riferimento")
                for n in taxonomy_norms
                if n.get("ref") or n.get("riferimento")
            ]
            kept_set = {r for r in kept_refs if r}
            discarded_refs = [
                n.get("ref") or n.get("riferimento")
                for n in (core_full + acc_full)
                if (n.get("ref") or n.get("riferimento"))
                and (n.get("ref") or n.get("riferimento")) not in kept_set
            ]
            self._log(
                f"🔎 [taxonomy] Causality {ct}: core {len(core_rel)}/{len(core_full)}, accessory {len(acc_rel)}/{len(acc_full)}"
            )
            if kept_refs:
                self._log(f"   ✔️ Kept: {', '.join(kept_refs)}")
            if discarded_refs:
                self._log(f"   ❌ Discarded: {', '.join(discarded_refs)}")

            # Align with Reasoner:
            # - taxonomy relevance (soft) from get_causality_theory_tool(claim=...)
            # - applicability: hard on accessory, soft on core
            core_pairs = [
                (n, self._norm_to_statute_dict(n))
                for n in core_rel
                if n.get("ref") or n.get("riferimento")
            ]
            acc_pairs = [
                (n, self._norm_to_statute_dict(n))
                for n in acc_rel
                if n.get("ref") or n.get("riferimento")
            ]
            core_statutes_raw = [st for _, st in core_pairs]
            acc_statutes_raw = [st for _, st in acc_pairs]
            (
                _core_applicable,
                acc_applicable,
                _core_rejected,
                _acc_rejected,
            ) = self._filter_taxonomy_anchor_statutes_by_applicability(
                claim,
                core_statutes=core_statutes_raw,
                accessory_statutes=acc_statutes_raw,
                log_prefix=f"counter/taxonomy/{ct}",
            )

            acc_keep_keys = {self._statute_identity_key(st) for st in acc_applicable}
            final_core_statutes = core_statutes_raw  # soft keep
            final_acc_statutes = [
                st
                for st in acc_statutes_raw
                if self._statute_identity_key(st) in acc_keep_keys
            ]

            for st in final_core_statutes:
                ref = str(st.get("statute_id") or "")
                if ref and ref not in seen_refs:
                    seen_refs.add(ref)
                    payload = dict(st)
                    payload["_kb_origin"] = "taxonomy_core"
                    statutes.append(payload)
            for st in final_acc_statutes:
                ref = str(st.get("statute_id") or "")
                if ref and ref not in seen_refs:
                    seen_refs.add(ref)
                    payload = dict(st)
                    payload["_kb_origin"] = "taxonomy_accessory"
                    statutes.append(payload)

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

        use_taxonomy_mode = False
        attack_source = "open"
        if enable_causality:
            # Select counter attacks from config pools
            taxonomy_selection = self._select_attacks(
                claim,
                routing_decision,
                reasoner_conclusion,
            )
            self._log(
                f"⚔️ Selected counter attacks: "
                f"{', '.join(taxonomy_selection.attack_ids) if taxonomy_selection.attack_ids else 'N/A'} "
                f"(pool size {len(taxonomy_selection.pool)})"
            )
            if taxonomy_selection.attack_ids:
                attack_selection = taxonomy_selection
                use_taxonomy_mode = True
                attack_source = "taxonomy"
            else:
                self._log(
                    "⚠️ No feasible taxonomy attacks; switching to open attack mode",
                    "warning",
                )
                attack_selection = self._select_open_attacks(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    min_attacks=2,
                    max_attacks=3,
                )
                use_taxonomy_mode = False
                attack_source = "open"
            if attack_selection.attack_ids:
                descs = [
                    f"{aid}: {attack_selection.descriptions.get(aid, 'N/A')}"
                    for aid in attack_selection.attack_ids
                ]
                self._log(f"📝 Attack descriptions: {' | '.join(descs)}")
            # Add filtered anchor norms only when taxonomy attacks are actually used.
            if use_taxonomy_mode:
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
                anchor_statutes = []
                boosted_counter_statutes = []
        else:
            self._log(
                "🔬 Causality DISABLED — using open attack mode (no taxonomy IDs)"
            )
            attack_selection = self._select_open_attacks(
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                min_attacks=2,
                max_attacks=3,
            )
            attack_source = "open"
            anchor_statutes = []
            boosted_counter_statutes = []

        if not attack_selection.attack_ids:
            self._log(
                "⚠️ No feasible counter attacks after strict filtering: abstaining",
                "warning",
            )
            return self._build_abstention_output(
                claim=claim,
                routing_decision=routing_decision,
                reasoner_conclusion=reasoner_conclusion,
                reason="no_feasible_attacks_after_filtering",
                relevant_statutes=pre_retrieved_statutes,
                relevant_precedents=pre_retrieved_precedents,
                attack_selection=attack_selection,
            )

        all_statutes = (
            pre_retrieved_statutes + boosted_counter_statutes + anchor_statutes
        )
        statute_origin_map: dict[tuple[str, str], set[str]] = {}
        for s in pre_retrieved_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add("pre_retrieval")
        for s in boosted_counter_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add("counter_second_pass")
        for s in anchor_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add(
                str(s.get("_kb_origin") or "taxonomy_anchor")
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
        before_expand_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        deduped_statutes = self._expand_with_cross_references(deduped_statutes)
        after_expand_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        for k in after_expand_keys - before_expand_keys:
            statute_origin_map.setdefault(k, set()).add("cross_ref")
        if len(deduped_statutes) > before_expand:
            self._log(
                f"➕ Added {len(deduped_statutes) - before_expand} statutes via cross-ref",
                "info",
            )
        self._log_final_statute_origins(
            deduped_statutes,
            statute_origin_map,
            label="Counter KB",
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
        target_map = self._extract_counter_target_map(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
        )
        MAX_CHAIN_RETRIES = settings.chain_max_retries
        output = None
        attack_blacklist: set[str] = set()

        for attempt in range(1, MAX_CHAIN_RETRIES + 1):
            self._log(
                f"🔄 Counter-Reasoner generation attempt {attempt}/{MAX_CHAIN_RETRIES}"
            )
            if attack_blacklist:
                self._log(
                    "⚠️ Counter attack blacklist active: "
                    + ", ".join(sorted(attack_blacklist)),
                    "warning",
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
                        target_map=target_map,
                        attack_blacklist=attack_blacklist,
                        stream_callback=stream_callback,
                    )
                )
            except Exception as gen_exc:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: counter planner/executor failed ({gen_exc})",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    return self._build_abstention_output(
                        claim=claim,
                        routing_decision=routing_decision,
                        reasoner_conclusion=reasoner_conclusion,
                        reason=f"generation_failed_after_retries: {gen_exc}",
                        relevant_statutes=deduped_statutes,
                        relevant_precedents=pre_retrieved_precedents,
                        attack_selection=attack_selection,
                    )
                continue

            resolved_attack_ids = [
                aid for aid in dict.fromkeys(step_attack_ids) if aid
            ] or list(attack_selection.attack_ids)
            resolved_primary_attack = (
                resolved_attack_ids[0]
                if resolved_attack_ids
                else attack_selection.attack_id
            )
            if any(aid.startswith("open_") for aid in resolved_attack_ids):
                attack_source = "open"

            # Build output
            output = CounterReasonerOutput(
                claim=claim,
                causal_type_id=routing_decision.causal_type_id,
                theory_id=routing_decision.theory_id,
                selected_attack_id=resolved_primary_attack,
                selected_attack_ids=resolved_attack_ids,
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
                    "selected_attack_id": resolved_primary_attack,
                    "selected_attack_ids": resolved_attack_ids,
                    "attack_source": attack_source,
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

    def _build_abstention_output(
        self,
        *,
        claim: str,
        routing_decision: RoutingDecision,
        reasoner_conclusion: str,
        reason: str,
        relevant_statutes: List[dict],
        relevant_precedents: List[dict],
        attack_selection: Optional["AttackSelection"] = None,
    ) -> CounterReasonerOutput:
        """Build a structured abstention output instead of raising hard errors."""
        raw_msg = (
            "Counter-Reasoner abstained: "
            f"{reason}. Non è stato possibile costruire una contro-catena valida "
            "senza contraddire i fatti espliciti o introdurre fatti nuovi."
        )
        return CounterReasonerOutput(
            claim=claim,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            selected_attack_id=(attack_selection.attack_id if attack_selection else ""),
            selected_attack_ids=(
                attack_selection.attack_ids if attack_selection else []
            ),
            reasoner_causality={
                "causal_type_id": routing_decision.causal_type_id,
                "theory_id": routing_decision.theory_id,
            },
            relevant_statutes=relevant_statutes,
            relevant_precedents=relevant_precedents,
            counter_arguments=[],
            reasoning_chain=[],
            raw_response=raw_msg,
            aspic_ir={
                "role": "counter",
                "reasoning_chain": [],
                "abstained": True,
                "abstention_reason": reason,
            },
            abstained=True,
            abstention_reason=reason,
            reasoner_conclusion_context=reasoner_conclusion,
        )

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

    def _estimate_counter_context_coverage(
        self,
        *,
        existing_statutes: List[dict],
        attack_ids: List[str],
    ) -> tuple[float, int]:
        """
        Heuristic estimate of how well current statutes cover selected counter attacks.

        Returns:
            (coverage_ratio, covered_attacks_count)
        """
        unique_attacks = [a for a in dict.fromkeys(attack_ids) if a]
        if not unique_attacks:
            return 1.0, 0
        if not existing_statutes:
            return 0.0, 0

        attack_lines = "\n".join(
            f"- {aid}: {self._attack_description(aid, locale='it', default=_DEFAULT_ATTACK_DESCRIPTION_IT)}"
            for aid in unique_attacks
        )

        statute_lines: List[str] = []
        for statute in existing_statutes[:18]:
            articolo = str(statute.get("articolo", "") or "")
            titolo = str(statute.get("titolo", "") or "")
            testo = re.sub(r"\s+", " ", str(statute.get("testo", "") or "")).strip()
            if len(testo) > 240:
                testo = f"{testo[:240]}..."
            statute_lines.append(f"- Art. {articolo} | {titolo} | {testo}")
        statutes_block = "\n".join(statute_lines) or "- none"

        prompt = (
            "Valuta la copertura del contesto normativo per i tipi di contro-attacco.\n\n"
            "TIPI DI ATTACCO (ID + descrizione):\n"
            f"{attack_lines}\n\n"
            "CONTESTO NORMATIVO DISPONIBILE:\n"
            f"{statutes_block}\n\n"
            "Compito:\n"
            "- indica per quali attack_id il contesto contiene almeno una norma idonea a sostenere quel tipo di attacco;\n"
            "- usa SOLO gli ID forniti.\n\n"
            'Rispondi SOLO in JSON compatto: {"covered_attack_ids":["id1","id2"]}'
        )

        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            raw = (resp.content or "").strip().replace("```json", "").replace("```", "")
            data = json.loads(raw)
            covered_raw = (
                data.get("covered_attack_ids", []) if isinstance(data, dict) else []
            )
            covered_ids = [
                str(x).strip()
                for x in covered_raw
                if str(x).strip() in set(unique_attacks)
            ]
            covered_ids = list(dict.fromkeys(covered_ids))
            covered = len(covered_ids)
            return covered / len(unique_attacks), covered
        except Exception as exc:
            self._log(
                f"⚠️ Counter coverage estimation failed (fallback quantitative): {exc}",
                "warning",
            )
            ratio = min(
                1.0,
                len(existing_statutes)
                / max(1, settings.counter_second_pass_min_against_statutes),
            )
            covered = min(len(unique_attacks), int(round(ratio * len(unique_attacks))))
            return ratio, covered

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

        coverage_ratio, covered_attacks = self._estimate_counter_context_coverage(
            existing_statutes=existing_statutes,
            attack_ids=attack_ids,
        )
        size_gate = (
            effective_context_size < settings.counter_second_pass_min_against_statutes
        )
        quality_gate = coverage_ratio < 0.50
        self._log(
            "Counter second-pass pre-check: "
            f"context_size={effective_context_size}, "
            f"attack_coverage={coverage_ratio:.2f} ({covered_attacks}/{max(1, len(set(attack_ids)))})",
            "info",
        )
        # Run second pass only when context is quantitatively small OR qualitatively
        # weak for the currently selected attacks.
        if not (size_gate or quality_gate):
            self._log(
                "Counter second-pass skipped: context already sufficient for selected attacks",
                "info",
            )
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
            pipe = get_legal_search_pipeline()
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

            legal_context = self._extract_legal_context(claim)
            for q_idx, query_text in enumerate(queries, start=1):
                embedding = pipe.embed_text(query_text)
                articles = pipe.vector_search(
                    embedding=embedding,
                    libri_filters=filters,
                    top_k=max(1, settings.counter_second_pass_top_k),
                    query_text=query_text,
                )
                article_by_id = {
                    (a.statute_id or "").strip(): a
                    for a in articles
                    if (a.statute_id or "").strip()
                }
                direct_candidates = []
                for article in articles:
                    statute_id = (article.statute_id or "").strip()
                    if (
                        not statute_id
                        or statute_id in existing_ids
                        or statute_id in boosted_by_id
                    ):
                        continue
                    payload = self._article_result_to_dict(article)
                    payload["_score"] = float(article.score)
                    direct_candidates.append(payload)

                if not direct_candidates:
                    continue

                direct_kept = self.filter_irrelevant_statutes(claim, direct_candidates)
                direct_kept = self.filter_applicable_statutes(
                    claim,
                    direct_kept,
                    legal_context,
                    cache_scope="counter_second_pass:direct",
                )

                for item in direct_kept:
                    statute_id = (item.get("statute_id") or "").strip()
                    if not statute_id:
                        continue
                    current = boosted_by_id.get(statute_id)
                    if current is None or float(item.get("_score", 0.0)) > float(
                        current.get("_score", 0.0)
                    ):
                        boosted_by_id[statute_id] = item

                seed_articles = [
                    article_by_id[(s.get("statute_id") or "").strip()]
                    for s in direct_kept
                    if (s.get("statute_id") or "").strip() in article_by_id
                ]
                cites_kept_count = 0
                if seed_articles:
                    expanded_articles = pipe.expand_with_cited_articles(seed_articles)
                    seed_ids = {
                        (a.statute_id or "").strip()
                        for a in seed_articles
                        if a.statute_id
                    }
                    cites_candidates = []
                    for article in expanded_articles:
                        statute_id = (article.statute_id or "").strip()
                        if (
                            not statute_id
                            or statute_id in seed_ids
                            or statute_id in existing_ids
                            or statute_id in boosted_by_id
                        ):
                            continue
                        payload = self._article_result_to_dict(article)
                        payload["_score"] = float(article.score)
                        cites_candidates.append(payload)
                    if cites_candidates:
                        cites_kept = self.filter_irrelevant_statutes(
                            claim, cites_candidates
                        )
                        cites_kept = self.filter_applicable_statutes(
                            claim,
                            cites_kept,
                            legal_context,
                            cache_scope="counter_second_pass:cites",
                        )
                        cites_kept_count = len(cites_kept)
                        for item in cites_kept:
                            statute_id = (item.get("statute_id") or "").strip()
                            if not statute_id:
                                continue
                            current = boosted_by_id.get(statute_id)
                            if current is None or float(
                                item.get("_score", 0.0)
                            ) > float(current.get("_score", 0.0)):
                                boosted_by_id[statute_id] = item

                self._log(
                    "📎 Counter second-pass query "
                    f"{q_idx}/{len(queries)}: direct_kept={len(direct_kept)}, "
                    f"cites_kept={cites_kept_count}",
                    "info",
                )

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
        target_map: Optional[Dict[str, List[str]]] = None,
        attack_blacklist: Optional[set[str]] = None,
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> tuple[str, List[str], List[str]]:
        """Generate counter-reasoning chain with plan -> execute -> residual replan workflow."""
        max_steps = settings.chain_max_steps
        min_steps = settings.chain_min_steps
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )
        target_map = target_map or {
            "allowed_targets": [],
            "forbidden_assumptions": [],
            "priority_targets": [],
        }
        target_map_text = self._target_map_text(target_map)

        selected_attack_ids = [
            aid
            for aid in dict.fromkeys(
                attack_selection.attack_ids or [attack_selection.attack_id]
            )
            if aid
        ]
        blacklist = attack_blacklist if attack_blacklist is not None else set()
        selected_attack_ids = [
            aid for aid in selected_attack_ids if aid not in blacklist
        ]
        primary_pool = [
            aid
            for aid in dict.fromkeys(attack_selection.pool)
            if aid and aid not in blacklist
        ]
        causal_pool = [
            aid
            for aid in dict.fromkeys(attack_selection.causal_pool or [])
            if aid and aid not in blacklist
        ]
        theory_pool = [
            aid
            for aid in dict.fromkeys(attack_selection.theory_pool or [])
            if aid and aid not in blacklist
        ]
        causal_residual = [aid for aid in causal_pool if aid not in primary_pool]
        theory_residual = [
            aid
            for aid in theory_pool
            if aid not in primary_pool and aid not in causal_residual
        ]
        backup_attack_ids = [
            aid
            for aid in dict.fromkeys(primary_pool + causal_residual + theory_residual)
            if aid and aid not in selected_attack_ids
        ]
        attack_desc_map: Dict[str, str] = dict(attack_selection.descriptions or {})
        for attack_id in selected_attack_ids + backup_attack_ids:
            attack_desc_map.setdefault(
                attack_id,
                self._attack_description(
                    attack_id,
                    locale="en",
                    default=_DEFAULT_ATTACK_DESCRIPTION_EN,
                ),
            )
        attack_fail_count: Dict[str, int] = {
            attack_id: 0 for attack_id in selected_attack_ids + backup_attack_ids
        }
        attack_failed_goals: Dict[str, set[str]] = {
            attack_id: set() for attack_id in selected_attack_ids + backup_attack_ids
        }
        attack_failure_patterns: Dict[str, Dict[str, int]] = {
            attack_id: defaultdict(int)
            for attack_id in selected_attack_ids + backup_attack_ids
        }
        attack_local_cooldown: Dict[str, int] = {}
        hard_goal_blacklist_threshold = 2

        def _goal_key(step: Dict[str, str]) -> str:
            novelty = str(step.get("novelty_key", "")).strip().lower()
            if novelty:
                return novelty
            goal = str(step.get("goal", "")).strip().lower()
            focus = str(step.get("focus", "")).strip().lower()
            key = re.sub(r"[^a-z0-9_]+", "_", f"{goal}_{focus}").strip("_")
            return key or "generic_goal"

        def _cooldown_blocked_ids() -> set[str]:
            return {
                aid
                for aid, rounds in attack_local_cooldown.items()
                if rounds > 0 and aid not in blacklist
            }

        def _rebuild_attack_catalog(current_ids: List[str]) -> str:
            ids = [aid for aid in current_ids if aid]
            if not ids:
                return "- no attacks available"
            return "\n".join(
                f"- {aid}: {attack_desc_map.get(aid, _DEFAULT_ATTACK_DESCRIPTION_EN)}"
                for aid in ids
            )

        attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
        steps: List[str] = []
        step_summaries: List[str] = []
        used_norms: List[str] = []
        step_attacks: List[str] = []
        allowed_statute_index = self._build_allowed_statute_index(available_statutes)
        plan_round = 0
        stalled_rounds = 0
        max_plan_rounds = max(1, self._max_plan_retries + 1)

        while len(steps) < min_steps and len(steps) < max_steps:
            plan_round += 1
            if plan_round > max_plan_rounds:
                break

            def _attack_is_active(aid: str) -> bool:
                return bool(aid and aid not in blacklist)

            selected_attack_ids = [
                aid
                for aid in selected_attack_ids
                if _attack_is_active(aid)
            ]
            backup_attack_ids = [
                aid
                for aid in backup_attack_ids
                if _attack_is_active(aid)
            ]

            if not selected_attack_ids:
                if backup_attack_ids:
                    replacement_idx = 0
                    blocked_ids = _cooldown_blocked_ids()
                    while replacement_idx < len(backup_attack_ids):
                        candidate = backup_attack_ids[replacement_idx]
                        if candidate not in blocked_ids:
                            selected_attack_ids.append(candidate)
                            backup_attack_ids.pop(replacement_idx)
                            break
                        replacement_idx += 1
                    if not selected_attack_ids and backup_attack_ids:
                        selected_attack_ids.append(backup_attack_ids.pop(0))
                    attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
                    self._log(
                        f"Warning: switching to backup attack {selected_attack_ids[0]}",
                        "warning",
                    )
                else:
                    open_selection = self._select_open_attacks(
                        claim=claim,
                        reasoner_conclusion=reasoner_conclusion,
                        min_attacks=1,
                        max_attacks=3,
                    )
                    open_ids = [
                        aid
                        for aid in dict.fromkeys(open_selection.attack_ids)
                        if aid and aid not in blacklist
                    ]
                    if not open_ids:
                        self._log(
                            "Warning: no additional feasible attacks available after rotations",
                            "warning",
                        )
                        break
                    selected_attack_ids = open_ids
                    for aid, desc in (open_selection.descriptions or {}).items():
                        if aid and desc:
                            attack_desc_map[aid] = desc
                    for aid in selected_attack_ids:
                        attack_fail_count.setdefault(aid, 0)
                        attack_failed_goals.setdefault(aid, set())
                        attack_failure_patterns.setdefault(aid, defaultdict(int))
                    attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
                    self._log(
                        "Warning: taxonomy attacks exhausted; switching to OPEN attack mode in-run",
                        "warning",
                    )

            remaining_min = max(1, min_steps - len(steps))
            remaining_max = max(1, max_steps - len(steps))
            planner_mode = "RESUME" if steps else "FULL"
            plan = self._generate_counter_plan(
                claim=claim,
                routing_decision=routing_decision,
                selected_attack_ids=selected_attack_ids,
                attack_catalog=attack_catalog,
                reasoner_conclusion=reasoner_conclusion,
                knowledge_base=knowledge_base,
                statutes_list=statutes_list,
                precedents_list=precedents_list,
                target_map_text=target_map_text,
                min_steps=remaining_min,
                max_steps=remaining_max,
                planner_mode=planner_mode,
                resume_from_step=len(steps) + 1,
                existing_summaries=step_summaries,
            )
            plan = self._prune_counter_plan_against_existing_history(
                plan=plan,
                previous_summaries=step_summaries,
            )
            plan = self._filter_counter_plan_by_feasibility(
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                target_map=target_map,
                plan=plan,
                previous_summaries=step_summaries,
            )
            if not plan:
                stalled_rounds += 1
                self._log(
                    "Warning: counter planner produced only redundant residual steps; retrying residual plan",
                    "warning",
                )
                if stalled_rounds >= 2:
                    break
                continue
            if len(plan) < remaining_min:
                stalled_rounds += 1
                self._log(
                    "Warning: counter planner returned too few feasible steps after pruning; replanning",
                    "warning",
                )
                if stalled_rounds >= 2:
                    break
                continue

            self._log(
                f"Counter plan generated: {len(plan)} step(s) "
                f"[round={plan_round}, mode={planner_mode}, completed={len(steps)}]"
            )

            steps_before_round = len(steps)
            round_failed = False

            for local_idx, plan_step in enumerate(plan, start=1):
                global_idx = len(steps) + 1
                blocked_ids = _cooldown_blocked_ids()
                step_attack_id = self._pick_attack_for_plan_step(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    plan_step=plan_step,
                    candidate_attack_ids=selected_attack_ids,
                    attack_desc_map=attack_desc_map,
                    attack_fail_count=attack_fail_count,
                    blocked_attack_ids=blocked_ids,
                )
                if not step_attack_id:
                    round_failed = True
                    self._log(
                        f"Warning: no attack available for planned counter-step {global_idx}; replanning",
                        "warning",
                    )
                    break
                plan_step["attack_id"] = step_attack_id
                step_attack_desc = attack_desc_map.get(
                    step_attack_id,
                    self._attack_description(
                        step_attack_id,
                        locale="en",
                        default=_DEFAULT_ATTACK_DESCRIPTION_EN,
                    ),
                )

                self._log(
                    f"Generating planned counter-step {global_idx}/{max_steps}: "
                    f"{plan_step.get('goal', '')[:80]} | attack={step_attack_id}"
                )
                (
                    step_text,
                    step_failure_reason,
                    hard_failure_count,
                ) = self._generate_counter_step_from_plan(
                    claim=claim,
                    routing_decision=routing_decision,
                    attack_id=step_attack_id,
                    attack_desc=step_attack_desc,
                    reasoner_conclusion=reasoner_conclusion,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    precedents_list=precedents_list,
                    plan=plan,
                    plan_index=local_idx,
                    plan_step=plan_step,
                    previous_steps=steps,
                    previous_summaries=step_summaries,
                    used_norms=used_norms,
                    allowed_statute_index=allowed_statute_index,
                    stream_callback=stream_callback,
                )
                if not step_text:
                    round_failed = True
                    attack_fail_count[step_attack_id] = (
                        attack_fail_count.get(step_attack_id, 0) + 1
                    )

                    hard_failure_detected = (
                        self._is_hard_attack_failure(step_failure_reason)
                        or hard_failure_count >= 2
                    )
                    if hard_failure_detected:
                        goal_key = _goal_key(plan_step)
                        goal_set = attack_failed_goals.setdefault(step_attack_id, set())
                        goal_set.add(goal_key)
                        pattern = self._normalize_failure_pattern(step_failure_reason)
                        pattern_map = attack_failure_patterns.setdefault(
                            step_attack_id, defaultdict(int)
                        )
                        pattern_map[pattern] += 1
                        same_pattern_hits = int(pattern_map.get(pattern, 0))
                        if (
                            same_pattern_hits >= 2
                            or len(goal_set) >= hard_goal_blacklist_threshold
                        ):
                            blacklist.add(step_attack_id)
                            attack_local_cooldown.pop(step_attack_id, None)
                            selected_attack_ids = [
                                aid for aid in selected_attack_ids if aid != step_attack_id
                            ]
                            backup_attack_ids = [
                                aid for aid in backup_attack_ids if aid != step_attack_id
                            ]
                            self._log(
                                f"Warning: rotating out attack {step_attack_id} "
                                f"after repeated hard failures "
                                f"(goals={len(goal_set)}, pattern={pattern}, "
                                f"pattern_hits={same_pattern_hits}, "
                                f"last_reason={step_failure_reason}, "
                                f"hard_attempts={hard_failure_count})",
                                "warning",
                            )
                        else:
                            # Temporary quarantine to avoid immediate reuse in next step/round.
                            attack_local_cooldown[step_attack_id] = max(
                                attack_local_cooldown.get(step_attack_id, 0), 2
                            )
                            if step_attack_id in selected_attack_ids:
                                selected_attack_ids = [
                                    aid
                                    for aid in selected_attack_ids
                                    if aid != step_attack_id
                                ] + [step_attack_id]
                            self._log(
                                f"Warning: quarantining attack {step_attack_id} "
                                f"after hard failure (goal={goal_key}, "
                                f"distinct_goals={len(goal_set)}; below blacklist threshold)",
                                "warning",
                            )

                    if backup_attack_ids and len(selected_attack_ids) < 2:
                        replacement = backup_attack_ids.pop(0)
                        blocked_ids = _cooldown_blocked_ids()
                        while replacement in blocked_ids and backup_attack_ids:
                            replacement = backup_attack_ids.pop(0)
                        if (
                            replacement not in selected_attack_ids
                            and replacement not in blacklist
                        ):
                            selected_attack_ids.append(replacement)
                            self._log(
                                f"Warning: injected backup attack {replacement} into active set",
                                "warning",
                            )
                            attack_failed_goals.setdefault(replacement, set())
                            attack_failure_patterns.setdefault(
                                replacement, defaultdict(int)
                            )

                    attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
                    self._log(
                        f"Planned counter step {global_idx} could not be generated; "
                        "replanning residual steps from accepted prefix "
                        f"(reason: {step_failure_reason or 'unknown'})",
                        "warning",
                    )
                    break

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
                prec_info = (
                    f" | prec: {', '.join(prec_mentions)}" if prec_mentions else ""
                )
                self._log(
                    f"Counter-step {global_idx}: {step_text[:80]}... "
                    f"| attack: {step_attack_id} "
                    f"| norms: {', '.join(new_norms) if new_norms else 'none'}{prec_info}"
                )

                if len(steps) >= max_steps:
                    break

            if len(steps) == steps_before_round:
                stalled_rounds += 1
            else:
                stalled_rounds = 0

            if attack_local_cooldown:
                for aid in list(attack_local_cooldown.keys()):
                    attack_local_cooldown[aid] = max(
                        0, int(attack_local_cooldown.get(aid, 0)) - 1
                    )
                    if attack_local_cooldown[aid] <= 0:
                        attack_local_cooldown.pop(aid, None)

            if stalled_rounds >= 2 and len(steps) < min_steps:
                break

            if not round_failed and len(steps) >= min_steps:
                break

        if len(steps) < min_steps:
            adaptive_min = 2 if min_steps >= 3 else 1
            if len(steps) >= adaptive_min:
                self._log(
                    "Warning: counter chain below chain_min_steps but accepted due low feasibility "
                    f"(generated={len(steps)}, configured_min={min_steps})",
                    "warning",
                )
            else:
                raise RuntimeError(
                    "Counter planner/executor produced fewer steps than chain_min_steps "
                    "after residual replanning"
                )

        self._log(
            f"Planned counter-chain complete: {len(steps)} steps, "
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
        target_map_text: str,
        min_steps: int,
        max_steps: int,
        planner_mode: str = "FULL",
        resume_from_step: int = 1,
        existing_summaries: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Generate and validate an execution plan for counter reasoning."""
        reasoner_block = f"\nReasoner conclusion to oppose:\n{reasoner_conclusion}\n"
        claim_facts = self._claim_fact_anchors_text(claim, max_items=10)
        existing_steps_text = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(existing_summaries or [], start=1)
            )
            if existing_summaries
            else "- none"
        )
        prompt = render_prompt(
            "counter_reasoner.generate_plan",
            claim=claim,
            reasoner_block=reasoner_block,
            claim_facts=claim_facts,
            routing_domain=routing_decision.domain,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            selected_attack_ids=", ".join(selected_attack_ids),
            attack_catalog=attack_catalog,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            knowledge_base=knowledge_base,
            target_map=target_map_text,
            min_steps=min_steps,
            max_steps=max_steps,
            planner_mode=planner_mode,
            resume_from_step=resume_from_step,
            existing_steps=existing_steps_text,
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
            step_type = self._normalize_counter_step_type(item.get("step_type"))
            novelty_key = (
                re.sub(
                    r"[^a-z0-9_]+",
                    "_",
                    str(item.get("novelty_key", "")).strip().lower(),
                ).strip("_")
                or re.sub(r"[^a-z0-9_]+", "_", focus.lower()).strip("_")[:48]
                or f"counter_step_{idx}"
            )
            citation_requirement = self._normalize_plan_citation_requirement(
                expected_norm=expected_norm,
                raw_value=item.get("citation_requirement"),
            )
            attack_id = str(item.get("attack_id", "")).strip()
            if not goal or not focus:
                continue
            if attack_id and attack_id not in allowed_attack_ids:
                attack_id = ""
            if not attack_id and allowed_attack_ids:
                # Keep plan attack-aware even when planner omits attack_id.
                attack_id = allowed_attack_ids[(idx - 1) % len(allowed_attack_ids)]
            cleaned.append(
                {
                    "id": str(item.get("id", f"C{idx}")).strip() or f"C{idx}",
                    "goal": goal,
                    "focus": focus,
                    "expected_norm": expected_norm,
                    "citation_requirement": citation_requirement,
                    "attack_id": attack_id,
                    "step_type": step_type,
                    "novelty_key": novelty_key[:64],
                }
            )

        if len(cleaned) < min_steps or len(cleaned) > max_steps:
            raise ValueError(
                f"invalid counter-plan length {len(cleaned)} (expected {min_steps}-{max_steps})"
            )
        novelty_keys = [step.get("novelty_key", "") for step in cleaned]
        if len(set(novelty_keys)) != len(novelty_keys):
            raise ValueError("counter planner produced duplicate novelty_key values")
        return cleaned

    def _pick_attack_for_plan_step(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        plan_step: Dict[str, str],
        candidate_attack_ids: List[str],
        attack_desc_map: Dict[str, str],
        attack_fail_count: Dict[str, int],
        blocked_attack_ids: Optional[set[str]] = None,
    ) -> str:
        """
        Pick an attack for a planned step.

        Disabled filters removed: no attack-alignment/compatibility/feasibility
        scoring is applied here.
        """
        _ = (claim, reasoner_conclusion, attack_desc_map, attack_fail_count)
        if not candidate_attack_ids:
            return ""
        blocked = {aid for aid in (blocked_attack_ids or set()) if aid}
        active_candidates = [
            aid for aid in candidate_attack_ids if aid and aid not in blocked
        ]
        if not active_candidates:
            # If all are temporarily blocked, fall back to the full set.
            active_candidates = list(candidate_attack_ids)
        if not active_candidates:
            return ""

        hinted = str(plan_step.get("attack_id", "")).strip()
        if hinted and hinted in active_candidates:
            return hinted

        return active_candidates[0]

    def _is_counter_plan_step_feasible(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        target_map: Dict[str, List[str]],
        plan_step: Dict[str, str],
        previous_summaries: List[str],
    ) -> tuple[bool, str]:
        """
        Lightweight feasibility gate for planner steps before step text generation.

        Important:
        - This gate must stay permissive.
        - Fact-lock/new-facts/opposition checks are validated on generated step text,
          not at planner-goal level.
        """
        goal = str(plan_step.get("goal", "")).strip()
        focus = str(plan_step.get("focus", "")).strip()
        if not goal or not focus:
            return False, "incomplete plan step"

        candidate = f"{goal}. {focus}".strip()
        if self._is_garbage_text(candidate, min_words=8):
            return False, "degenerate plan step"
        if previous_summaries and self._is_repetitive_step(
            candidate, previous_summaries, threshold=0.45
        ):
            return False, "plan step redundant with accepted history"
        _ = (claim, reasoner_conclusion, target_map)
        return True, ""

    def _filter_counter_plan_by_feasibility(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        target_map: Dict[str, List[str]],
        plan: List[Dict[str, str]],
        previous_summaries: List[str],
    ) -> List[Dict[str, str]]:
        """
        Drop planner steps that are impossible under fact-lock/opposition constraints.
        """
        if not plan:
            return []
        kept: List[Dict[str, str]] = []
        dropped = 0
        for step in plan:
            ok, _ = self._is_counter_plan_step_feasible(
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                target_map=target_map,
                plan_step=step,
                previous_summaries=previous_summaries,
            )
            if ok:
                kept.append(step)
            else:
                dropped += 1
        if dropped:
            self._log(
                f"Warning: counter plan feasibility gate pruned {dropped} step(s)",
                "warning",
            )
        return kept

    @staticmethod
    def _normalize_counter_step_type(raw_value: object) -> str:
        """Normalize counter planner step_type into a constrained enum-like value."""
        value = str(raw_value or "").strip().upper()
        allowed = {
            "TARGET_FACTS",
            "TARGET_CAUSAL_LINK",
            "TARGET_LEGAL_QUALIFICATION",
            "TARGET_ELEMENT",
            "TARGET_BALANCING",
            "TARGET_OUTCOME",
            "OTHER",
        }
        if value in allowed:
            return value
        aliases = {
            "FACTS": "TARGET_FACTS",
            "CAUSAL_LINK": "TARGET_CAUSAL_LINK",
            "QUALIFICATION": "TARGET_LEGAL_QUALIFICATION",
            "ELEMENTS": "TARGET_ELEMENT",
            "BALANCING": "TARGET_BALANCING",
            "OUTCOME": "TARGET_OUTCOME",
            "CONSEQUENCE": "TARGET_OUTCOME",
        }
        return aliases.get(value, "OTHER")

    @staticmethod
    def _has_min_counter_plan_type_coverage(plan_steps: List[Dict[str, str]]) -> bool:
        """Require minimal diversity of counter step types for non-trivial plans."""
        if len(plan_steps) <= 2:
            return True
        concrete = [
            str(step.get("step_type", "")).strip().upper()
            for step in plan_steps
            if str(step.get("step_type", "")).strip().upper() not in {"", "OTHER"}
        ]
        if len(concrete) < 2:
            return True
        min_required = 2 if len(plan_steps) <= 4 else 3
        return len(set(concrete)) >= min_required

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
        """Detect overlap across planned goals/focuses (lexical + semantic)."""
        normalized = []
        texts: List[str] = []
        for step in plan_steps:
            text = f"{step.get('goal', '')} {step.get('focus', '')}".lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            words = {w for w in text.split() if len(w) > 3}
            normalized.append(words)
            texts.append(
                re.sub(
                    r"\s+",
                    " ",
                    f"{step.get('goal', '')}. {step.get('focus', '')}",
                ).strip()
            )

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                a = normalized[i]
                b = normalized[j]
                if not a or not b:
                    continue
                overlap = len(a & b) / len(a | b)
                if overlap >= 0.65:
                    return True
                if overlap >= 0.40:
                    rel_ab = self._nli_relation(
                        target_text=texts[i],
                        attacker_text=texts[j],
                        actor_label="CounterPlanner",
                    )
                    rel_ba = self._nli_relation(
                        target_text=texts[j],
                        attacker_text=texts[i],
                        actor_label="CounterPlanner",
                    )
                    if rel_ab == "entailment" and rel_ba == "entailment":
                        return True
        return False

    def _prune_counter_plan_against_existing_history(
        self,
        *,
        plan: List[Dict[str, str]],
        previous_summaries: List[str],
    ) -> List[Dict[str, str]]:
        """Remove residual counter-plan steps already covered by accepted prefix."""
        if not plan or not previous_summaries:
            return plan
        pruned: List[Dict[str, str]] = []
        removed = 0
        for step in plan:
            candidate = f"{step.get('goal', '')}. {step.get('focus', '')}".strip()
            if not candidate:
                removed += 1
                continue
            if self._is_repetitive_step(candidate, previous_summaries, threshold=0.45):
                removed += 1
                continue
            redundant = False
            for summary in previous_summaries[-4:]:
                rel = self._nli_relation(
                    target_text=summary,
                    attacker_text=candidate,
                    actor_label="CounterPlanner",
                )
                if rel == "entailment":
                    rel_sym = self._nli_relation(
                        target_text=candidate,
                        attacker_text=summary,
                        actor_label="CounterPlanner",
                    )
                    if rel_sym in {"entailment", "neutral"}:
                        redundant = True
                        break
            if redundant:
                removed += 1
                continue
            pruned.append(step)
        if removed:
            self._log(
                f"Warning: counter planner normalization pruned {removed} residual step(s) already covered by accepted prefix",
                "warning",
            )
        return pruned

    @staticmethod
    def _is_hard_attack_failure(reason: str) -> bool:
        """
        Identify failure causes that indicate the current attack line is structurally unproductive.
        """
        reason_norm = (reason or "").strip().lower()
        if not reason_norm:
            return False
        hard_markers = (
            "adds factual allegations not present in claim",
            "contradicts explicit claim fact",
            "candidate contradicts previous step",
            "history incompatibility",
            "citation not grounded in allowed statutes",
            "contains ungrounded citation",
        )
        return any(marker in reason_norm for marker in hard_markers)

    @staticmethod
    def _normalize_failure_pattern(reason: str) -> str:
        """Map raw reject reasons to a stable failure-pattern label."""
        text = (reason or "").strip().lower()
        if "adds factual allegations" in text:
            return "adds_facts"
        if "contradicts explicit claim fact" in text:
            return "contradicts_claim_fact"
        if "candidate contradicts previous step" in text:
            return "history_contradiction"
        if "history incompatibility" in text:
            return "history_incompatibility"
        if "citation not grounded" in text or "ungrounded citation" in text:
            return "ungrounded_citation"
        if "lexical repetition" in text:
            return "repetition"
        if "generation error" in text:
            return "generation_error"
        return "other"

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
    ) -> tuple[str, str, int]:
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
        hard_failure_count = 0
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
            except PipelineCancelled:
                raise
            except Exception as exc:
                last_reason = f"generation error: {exc}"
                if self._is_hard_attack_failure(last_reason):
                    hard_failure_count += 1
                    if hard_failure_count >= 2:
                        self._log(
                            f"⚠️ Counter-step {plan_index}: hard-failure threshold reached during generation "
                            f"(hard_attempts={hard_failure_count}), rotating attack immediately",
                            "warning",
                        )
                        return "", last_reason, hard_failure_count
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
                attack_id=(plan_step.get("attack_id") or attack_id),
                attack_desc=attack_desc,
                plan_focus=plan_step.get("focus", ""),
            )
            if ok:
                return candidate, "", 0
            last_reason = reason
            if self._is_hard_attack_failure(reason):
                hard_failure_count += 1
                if hard_failure_count >= 2:
                    self._log(
                        f"⚠️ Counter-step {plan_index}: hard-failure threshold reached "
                        f"(hard_attempts={hard_failure_count}), rotating attack immediately",
                        "warning",
                    )
                    return "", last_reason, hard_failure_count
            if stream_callback:
                try:
                    stream_callback(
                        {
                            "phase": "counter",
                            "action": "reset_step",
                            "step": plan_index,
                        }
                    )
                except PipelineCancelled:
                    raise
                except Exception:
                    pass
            self._log(
                f"⚠️ Counter-step {plan_index} rejected ({reason}) "
                f"[attempt {attempt}/{self._max_step_rewrites + 1}]",
                "warning",
            )
        return "", last_reason, hard_failure_count

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
            f"{idx}. {step.get('goal', '')} | focus: {step.get('focus', '')} | "
            f"attack: {step.get('attack_id', '')} | type: {step.get('step_type', 'OTHER')} | "
            f"novelty: {step.get('novelty_key', '')}"
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
        claim_facts = self._claim_fact_anchors_text(claim, max_items=10)
        return render_prompt(
            "counter_reasoner.step_prompt",
            claim=claim,
            reasoner_block=reasoner_block,
            claim_facts=claim_facts,
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
            plan_attack_id=(plan_step.get("attack_id") or attack_id),
            plan_step_type=plan_step.get("step_type", "OTHER"),
            plan_novelty_key=plan_step.get("novelty_key", ""),
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
        facts_ok, facts_reason = self._is_counter_step_fact_consistent_with_claim(
            claim=claim,
            candidate_step=text,
        )
        if not facts_ok:
            return False, facts_reason
        grounded_ok, grounded_reason = self._is_counter_step_grounded_in_claim_facts(
            claim=claim,
            candidate_step=text,
        )
        if not grounded_ok:
            return False, grounded_reason
        compatible, compat_reason = self._is_counter_step_compatible_with_history(
            candidate_step=text,
            previous_steps=previous_steps,
            claim=claim,
        )
        if not compatible:
            return False, compat_reason or "history incompatibility"
        _ = (reasoner_conclusion, attack_id, attack_desc, plan_focus)
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
        if previous_steps and self._is_repetitive_step(text, previous_steps):
            return False, "lexical repetition"
        return True, ""

    def _is_counter_step_compatible_with_history(
        self,
        *,
        candidate_step: str,
        previous_steps: List[str],
        claim: str,
    ) -> tuple[bool, str]:
        """
        Ensure candidate counter-step is semantically compatible with accepted chain history.
        """
        _ = claim
        if not previous_steps:
            return True, ""
        window = previous_steps[-3:]
        start_idx = len(previous_steps) - len(window) + 1
        for offset, prev_step in enumerate(window):
            relation = self._nli_relation(
                target_text=prev_step,
                attacker_text=candidate_step,
                actor_label="CounterReasoner",
            )
            if relation != "contradiction":
                continue
            # Reject only when contradiction is stable in both directions.
            relation_sym = self._nli_relation(
                target_text=candidate_step,
                attacker_text=prev_step,
                actor_label="CounterReasoner",
            )
            if relation_sym == "contradiction":
                step_no = start_idx + offset
                return False, f"candidate contradicts previous step {step_no}"
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

    def _is_counter_step_grounded_in_claim_facts(
        self,
        *,
        claim: str,
        candidate_step: str,
    ) -> tuple[bool, str]:
        """
        Reject counter-steps that add new factual allegations not present in the claim.

        Legal inferences from existing claim facts are allowed.
        """
        claim_text = (claim or "").strip()
        step_text = (candidate_step or "").strip()
        if not claim_text or not step_text:
            return True, ""

        cache_key = (claim_text, step_text)
        cached = self._new_facts_check_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "counter_reasoner.no_new_facts",
            claim=claim_text,
            candidate_step=step_text,
        )
        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            answer = (resp.content or "").strip().upper()
            if "ADDS_FACTS" in answer:
                result = (
                    False,
                    "adds factual allegations not present in claim",
                )
            elif "LEGAL_INFERENCE" in answer:
                result = (True, "")
            else:
                result = (True, "")
        except Exception as exc:
            # Avoid over-blocking on checker outages/rate limits.
            self._log(
                f"⚠️ Counter new-facts check failed (fallback keep): {exc}",
                "warning",
            )
            result = (True, "")

        self._new_facts_check_cache[cache_key] = result
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

    @staticmethod
    def _first_sentence_legal_safe(text: str) -> str:
        """
        Return first sentence without splitting on legal abbreviations
        such as ``art.``, ``c.p.``, ``c.c.``, ``c.p.p.``.
        """
        if not text:
            return ""
        protected = str(text)
        replacements = [
            (r"\bc\.p\.p\.", "CPPTOKEN"),
            (r"\bc\.p\.", "CPTOKEN"),
            (r"\bc\.c\.", "CCTOKEN"),
            (r"\bart\.", "ARTTOKEN"),
            (r"\bd\.lgs\.", "DLGSTOKEN"),
            (r"\bd\.p\.r\.", "DPRTOKEN"),
            (r"\bn\.", "NTOKEN"),
            (r"\bl\.", "LTOKEN"),
        ]
        for pattern, token in replacements:
            protected = re.sub(pattern, token, protected, flags=re.IGNORECASE)

        first = re.split(r"(?<=[.!?])\s+", protected.strip())[0]

        restores = [
            ("CPPTOKEN", "c.p.p."),
            ("CPTOKEN", "c.p."),
            ("CCTOKEN", "c.c."),
            ("ARTTOKEN", "art."),
            ("DLGSTOKEN", "d.lgs."),
            ("DPRTOKEN", "d.p.r."),
            ("NTOKEN", "n."),
            ("LTOKEN", "l."),
        ]
        for token, value in restores:
            first = first.replace(token, value)
        return first.strip()

    @staticmethod
    def _compact_step_summary(step_text: str) -> str:
        """Compact summary used as execution memory for following steps."""
        first_sentence = CounterReasoner._first_sentence_legal_safe(step_text)
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
        except PipelineCancelled:
            raise
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

    def _derive_counter_conclusion_ground(self, steps: List[str]) -> str:
        """Pick a final counter-rationale from the last coherent step."""
        for step in reversed(steps):
            first_sentence = self._first_sentence_legal_safe(step)
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
    causal_pool: List[str] = field(default_factory=list)
    theory_pool: List[str] = field(default_factory=list)

    @property
    def attack_id(self) -> str:
        """Primary attack id (first selected), kept for compatibility."""
        return self.attack_ids[0] if self.attack_ids else ""

    @property
    def description(self) -> str:
        """Primary attack description (first selected), kept for compatibility."""
        return self.descriptions.get(self.attack_id, "")
