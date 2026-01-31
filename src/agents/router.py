"""
Pre-routing component that selects causal_type_id and theory_id before reasoning.

The router is lightweight and uses the causal configuration as the single source
of truth. It validates the LLM choice against the config and exposes anchor
norms and principle tests for downstream agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from .base import AgentConfig, BaseAgent
from .tools import config_loader

ROUTER_SYSTEM_PROMPT = """You are a preliminary router for a causal reasoning system.
Your only task is to assign:
- a `causal_type_id` (choose ONLY from the listed ids)
- a `theory_id` applicable to that causal_type_id (choose ONLY from the listed ids)

Rules:
- Respond ONLY with compact JSON: {"causal_type_id": "...", "theory_id": "..."}
- Do not add text, comments, or explanations.
- If uncertain, pick the suggested default pair.
"""


@dataclass
class RoutingDecision:
    """Structured output of the router."""

    claim: str
    causal_type_id: str
    theory_id: str
    anchor_norms: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    principle_tests: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causal_type_id": self.causal_type_id,
            "theory_id": self.theory_id,
            "anchor_norms": self.anchor_norms,
            "principle_tests": self.principle_tests,
        }


class Router(BaseAgent):
    """
    Lightweight router that classifies the claim using config_taxonomy.json.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self._config = config_loader.load_config()
        self._causal_types = config_loader.causal_types_by_id(self._config)
        self._theories = config_loader.theories_by_id(self._config)
        self._defaults = config_loader.default_mapping_by_causal(self._config)

    def route(self, claim: str) -> RoutingDecision:
        """Route a claim to (causal_type_id, theory_id)."""
        self._log(f"🔀 Routing claim: {claim[:80]}...")
        candidate = self._route_with_llm(claim)

        validated_causal, validated_theory = config_loader.validate_ids(
            candidate.get("causal_type_id", ""),
            candidate.get("theory_id"),
            self._config,
        )

        anchor_norms = config_loader.anchor_norms_for(validated_causal, self._config)
        principle_tests = config_loader.principle_tests_for(
            validated_causal, self._config
        )

        self._log(
            f"🎯 Router decision -> causal_type_id={validated_causal}, theory_id={validated_theory}"
        )

        return RoutingDecision(
            claim=claim,
            causal_type_id=validated_causal,
            theory_id=validated_theory or "",
            anchor_norms=anchor_norms,
            principle_tests=principle_tests,
        )

    # BaseAgent abstract method compatibility
    def run(self, claim: str, *args, **kwargs) -> RoutingDecision:  # type: ignore[override]
        """Alias for route() to satisfy BaseAgent interface."""
        return self.route(claim)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _route_with_llm(self, claim: str) -> Dict[str, str]:
        """Use LLM to pick ids, falling back to defaults if parsing fails."""
        prompt = self._build_prompt(claim)
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = (resp.content or "").strip()
            parsed = self._parse_json_like(content)
            if parsed:
                return parsed
        except Exception as e:
            self._log(f"⚠️ Router LLM fallito: {e}", "warning")

        # Fallback to first default mapping
        if self._defaults:
            first_ct, mapping = next(iter(self._defaults.items()))
            return {
                "causal_type_id": first_ct,
                "theory_id": mapping.get("reasoner_primary_theory")
                or config_loader.pick_default_theory(first_ct, self._config)
                or "",
            }
        return {"causal_type_id": "", "theory_id": ""}

    def _build_prompt(self, claim: str) -> str:
        """Build a compact instruction listing available ids."""
        type_lines = []
        for ct_id, ct in self._causal_types.items():
            default_th = self._defaults.get(ct_id, {}).get(
                "reasoner_primary_theory", ""
            )
            type_lines.append(
                f"- {ct_id}: {ct.get('name', '')} [{ct.get('domain', '')}] | default theory: {default_th}"
            )

        theory_lines = []
        for th_id, th in self._theories.items():
            theory_lines.append(
                f"- {th_id}: {th.get('name','')} | applicabile a: {', '.join(th.get('applicable_causal_types', []))}"
            )

        return f"""{ROUTER_SYSTEM_PROMPT}

Claim:
\"\"\"{claim}\"\"\"

Opzioni causal_type_id:
{chr(10).join(type_lines)}

Opzioni theory_id:
{chr(10).join(theory_lines)}

Ricorda: scegli SOLO id validi e restituisci JSON compatto."""

    def _parse_json_like(self, text: str) -> Dict[str, str]:
        """Parse JSON, tolerating fenced code."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return {
                    "causal_type_id": str(data.get("causal_type_id", "")).strip(),
                    "theory_id": str(data.get("theory_id", "")).strip(),
                }
        except Exception:
            pass
        return {}
