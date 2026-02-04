"""ASPIC-style IR formatter.

ID legend:
- A#  argument
- P#  premise
- R#  rule
- C#  conclusion
- S#  reasoning chain step
- PR# precedent node (linked to steps/argument parts when cited)
"""

from __future__ import annotations

import re
from typing import Any

_SECTION_ALIASES = [
    ("premessa alternativa", "premise"),
    ("premessa", "premise"),
    ("norma", "norm"),
    ("causal link", "link"),
    ("nesso causale alternativo", "link"),
    ("nesso causale", "link"),
    ("nesso", "link"),
    ("conclusione contraria", "conclusion"),
    ("conclusione", "conclusion"),
]

_ARTICLE_PATTERN = re.compile(
    r"(?:art(?:icolo|icoli)?\.?\s*)(\d{1,4}(?:-[a-z0-9]+)?)\s*"
    r"(c\.?c\.?|c\.?p\.?|codice\s+civile|codice\s+penale)?",
    re.IGNORECASE,
)

_ARTICLE_LIST_PATTERN = re.compile(
    r"articoli?\s+([0-9,\s/\-e]+)\s*"
    r"(c\.?c\.?|c\.?p\.?|codice\s+civile|codice\s+penale)",
    re.IGNORECASE,
)


class AspicFormatter:
    def __init__(
        self,
        role: str,
        statutes: list[dict] | None = None,
        precedents: list[dict] | None = None,
    ):
        self.role = role
        self._statute_index, self._statute_by_num = _build_statute_index(statutes or [])
        self._precedents = precedents or []

    def format(
        self,
        *,
        claim: str,
        raw_response: str,
        reasoning_chain: list[str],
        arguments: list[Any] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        chain_steps = _build_chain_steps(
            reasoning_chain, self._statute_index, self._statute_by_num, self._precedents
        )
        blocks = _parse_argument_blocks(raw_response)
        if not blocks and arguments:
            blocks = _blocks_from_arguments(arguments)
        if not blocks:
            blocks = _blocks_from_chain(chain_steps)

        arguments_ir = _build_arguments_ir(
            blocks,
            chain_steps,
            self._statute_index,
            self._statute_by_num,
            self._precedents,
            role=self.role,
        )

        sources = _build_sources(self._statute_index, self._precedents)
        precedent_nodes, precedent_links = _build_precedent_graph(
            chain_steps, arguments_ir, self._precedents
        )

        return {
            "schema": "aspic_ir_v1",
            "role": self.role,
            "claim": claim,
            "raw_response": raw_response,
            "reasoning_chain": chain_steps,
            "arguments": arguments_ir,
            "sources": sources,
            "precedent_nodes": precedent_nodes,
            "precedent_links": precedent_links,
            "metadata": metadata or {},
        }


def _strip_markup(line: str) -> str:
    cleaned = line.strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.strip("*")
    cleaned = cleaned.lstrip("-* ")
    return cleaned.strip()


def _normalize_article_number(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^art(?:icolo|icoli)?\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"c\.?\s*c\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"c\.?\s*p\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"codice\s+civile", "", text, flags=re.IGNORECASE)
    text = re.sub(r"codice\s+penale", "", text, flags=re.IGNORECASE)
    return text.strip().strip(".").strip()


def _is_argument_header(text: str) -> bool:
    lower = text.strip().lower()
    return lower.startswith("argomento") or lower.startswith("argument")


def _is_chain_header(text: str) -> bool:
    lower = text.strip().lower()
    return lower.startswith("chain of reasoning") or lower.startswith(
        "catena di ragionamento"
    )


def _is_chain_noise(text: str) -> bool:
    lower = text.strip().lower()
    return lower.startswith(
        (
            "argomento",
            "argument",
            "norma",
            "testo",
            "causal link",
            "nesso causale",
            "premessa",
            "conclusione",
            "chain of reasoning",
            "catena di ragionamento",
            "nota",
        )
    )


def _detect_section(line: str) -> tuple[str, str]:
    cleaned = _strip_markup(line)
    lower = cleaned.lower()
    for label, key in _SECTION_ALIASES:
        if lower.startswith(label):
            content = cleaned[len(label) :].lstrip(" :.-")
            return key, content
    return "", ""


def _parse_argument_blocks(raw_response: str) -> list[dict]:
    blocks: list[dict[str, str]] = []
    current = {"premise": "", "norm": "", "link": "", "conclusion": ""}
    active_label = ""
    seen_any = False
    global_premise_lines: list[str] = []
    seen_argument_header = False
    in_global_premise = False

    for raw_line in raw_response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = _strip_markup(line)
        if not cleaned:
            continue
        if _is_chain_header(cleaned):
            break
        if _is_argument_header(cleaned):
            if any(current.values()):
                blocks.append(current)
                current = {"premise": "", "norm": "", "link": "", "conclusion": ""}
            active_label = ""
            seen_argument_header = True
            in_global_premise = False
            continue
        if in_global_premise:
            label, content = _detect_section(line)
            if label and label != "premise":
                in_global_premise = False
            else:
                global_premise_lines.append(content or cleaned)
                continue
        if active_label and re.match(r"^\d+[\.)]\s+", line):
            active_label = ""
            continue
        label, content = _detect_section(line)
        if label:
            if (
                label == "premise"
                and not seen_argument_header
                and not any(current.values())
            ):
                in_global_premise = True
                if content:
                    global_premise_lines.append(content)
                continue
            if label == "premise" and seen_any and any(current.values()):
                blocks.append(current)
                current = {"premise": "", "norm": "", "link": "", "conclusion": ""}
            active_label = label
            seen_any = True
            if content:
                current[label] = (current[label] + " " + content).strip()
            continue
        if active_label:
            current[active_label] = (current[active_label] + " " + line).strip()

    if seen_any and any(current.values()):
        blocks.append(current)

    global_premise = " ".join(global_premise_lines).strip()
    if global_premise:
        for block in blocks:
            if block.get("premise"):
                block["premise"] = f"{global_premise} {block['premise']}".strip()
            else:
                block["premise"] = global_premise

    return blocks


def _blocks_from_arguments(arguments: list[Any]) -> list[dict]:
    blocks: list[dict[str, str]] = []
    for arg in arguments:
        if hasattr(arg, "premise"):
            blocks.append(
                {
                    "premise": _strip_markup(getattr(arg, "premise", "") or ""),
                    "norm": _strip_markup(getattr(arg, "norm", "") or ""),
                    "link": _strip_markup(getattr(arg, "link", "") or ""),
                    "conclusion": _strip_markup(getattr(arg, "conclusion", "") or ""),
                }
            )
            continue
        if isinstance(arg, dict):
            blocks.append(
                {
                    "premise": _strip_markup(
                        arg.get("premise")
                        or arg.get("premessa")
                        or arg.get("premise_text")
                        or ""
                    ),
                    "norm": _strip_markup(arg.get("norm") or arg.get("norma") or ""),
                    "link": _strip_markup(arg.get("link") or arg.get("nesso") or ""),
                    "conclusion": _strip_markup(
                        arg.get("conclusion") or arg.get("conclusione") or ""
                    ),
                }
            )
    return [b for b in blocks if any(b.values())]


def _blocks_from_chain(chain_steps: list[dict]) -> list[dict]:
    if not chain_steps:
        return []
    if len(chain_steps) == 1:
        return [
            {
                "premise": "",
                "norm": "",
                "link": "",
                "conclusion": chain_steps[0]["text"],
            }
        ]
    premises = " ".join(step["text"] for step in chain_steps[:-1])
    return [
        {
            "premise": premises,
            "norm": "",
            "link": "",
            "conclusion": chain_steps[-1]["text"],
        }
    ]


def _build_statute_index(statutes: list[dict]) -> tuple[dict, dict]:
    index: dict[tuple[str, str], dict] = {}
    by_num: dict[str, dict] = {}
    for s in statutes:
        num_raw = str(s.get("articolo", "")).strip()
        num = _normalize_article_number(num_raw)
        if not num:
            continue
        source = (s.get("source") or "").strip()
        label = _statute_label(num, source)
        info = {
            "statute_id": s.get("statute_id") or label,
            "articolo": num,
            "source": source,
            "label": label,
            "title": s.get("titolo") or "",
        }
        index[(num, source)] = info
        by_num.setdefault(num, info)
    return index, by_num


def _statute_label(num: str, source: str) -> str:
    if source == "codice_civile":
        return f"Art. {num} c.c."
    if source == "codice_penale":
        return f"Art. {num} c.p."
    return f"Art. {num}"


def _normalize_source_from_code(code: str) -> str:
    if not code:
        return ""
    code = code.lower().strip()
    if "civ" in code or "c.c" in code:
        return "codice_civile"
    if "pen" in code or "c.p" in code:
        return "codice_penale"
    return ""


def _extract_citations(
    text: str,
    statute_index: dict,
    statute_by_num: dict,
    precedents: list[dict],
) -> dict:
    statutes: list[dict] = []
    unknown_statutes: list[dict] = []
    seen_pairs = set()

    list_matches = _ARTICLE_LIST_PATTERN.findall(text)
    for list_text, code in list_matches:
        source = _normalize_source_from_code(code)
        for raw_num in re.findall(r"\d{1,4}(?:-[a-z0-9]+)?", list_text, re.IGNORECASE):
            num = _normalize_article_number(raw_num)
            if not num:
                continue
            pair = (num, source)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            info = None
            if source and (num, source) in statute_index:
                info = statute_index[(num, source)]
            elif num in statute_by_num:
                info = statute_by_num[num]
            if info:
                statutes.append(info)
            else:
                unknown_statutes.append({"articolo": num, "source": source})

    for raw_num, code in _ARTICLE_PATTERN.findall(text):
        num = _normalize_article_number(raw_num)
        source = _normalize_source_from_code(code)
        if not num:
            continue
        pair = (num, source)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        info = None
        if source and (num, source) in statute_index:
            info = statute_index[(num, source)]
        elif num in statute_by_num:
            info = statute_by_num[num]
        if info:
            statutes.append(info)
        else:
            unknown_statutes.append({"articolo": num, "source": source})

    precedent_refs: list[dict] = []
    lower_text = text.lower()
    for p in precedents:
        title = (p.get("title") or "").strip()
        if title and title.lower() in lower_text:
            precedent_refs.append(
                {
                    "precedent_id": p.get("precedent_id") or title,
                    "title": title,
                }
            )

    return {
        "statutes": _dedup_list(statutes, key="statute_id"),
        "precedents": _dedup_list(precedent_refs, key="precedent_id"),
        "unknown_statutes": _dedup_list(unknown_statutes, key="articolo"),
    }


def _build_chain_steps(
    reasoning_chain: list[str],
    statute_index: dict,
    statute_by_num: dict,
    precedents: list[dict],
) -> list[dict]:
    steps: list[dict] = []
    seen_texts = set()
    for step in reasoning_chain:
        text = _strip_markup(step)
        if not text:
            continue
        if _is_chain_noise(text):
            continue
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if normalized in seen_texts:
            continue
        if ":" in text:
            after = text.split(":", 1)[1].strip()
            after_norm = re.sub(r"\s+", " ", after.lower()).strip()
            if after_norm in seen_texts:
                continue
        step_id = f"S{len(steps) + 1}"
        seen_texts.add(normalized)
        steps.append(
            {
                "id": step_id,
                "text": text,
                "citations": _extract_citations(
                    text, statute_index, statute_by_num, precedents
                ),
            }
        )
    return steps


def _build_arguments_ir(
    blocks: list[dict],
    chain_steps: list[dict],
    statute_index: dict,
    statute_by_num: dict,
    precedents: list[dict],
    *,
    role: str,
) -> list[dict]:
    arguments: list[dict] = []
    for idx, block in enumerate(blocks, start=1):
        arg_id = f"A{idx}"
        premises = []
        if block.get("premise"):
            premises.append(
                {
                    "id": f"{arg_id}.P1",
                    "type": "premise",
                    "text": block["premise"],
                    "citations": _extract_citations(
                        block["premise"], statute_index, statute_by_num, precedents
                    ),
                }
            )
        if block.get("norm"):
            premises.append(
                {
                    "id": f"{arg_id}.P2",
                    "type": "norm",
                    "text": block["norm"],
                    "citations": _extract_citations(
                        block["norm"], statute_index, statute_by_num, precedents
                    ),
                }
            )

        rule_text = block.get("link", "")
        rule = (
            {
                "id": f"{arg_id}.R1",
                "type": "defeasible",
                "text": rule_text,
            }
            if rule_text
            else {}
        )

        conclusion_text = block.get("conclusion") or ""
        if not conclusion_text and chain_steps:
            conclusion_text = chain_steps[-1]["text"]
        conclusion = (
            {
                "id": f"{arg_id}.C1",
                "text": conclusion_text,
                "citations": _extract_citations(
                    conclusion_text, statute_index, statute_by_num, precedents
                ),
            }
            if conclusion_text
            else {}
        )

        arguments.append(
            {
                "id": arg_id,
                "role": role,
                "premises": premises,
                "rule": rule,
                "conclusion": conclusion,
            }
        )
    return arguments


def _build_sources(statute_index: dict, precedents: list[dict]) -> dict:
    statutes = [info for info in statute_index.values()]
    precedents_summary = []
    for p in precedents:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        precedents_summary.append(
            {
                "precedent_id": p.get("precedent_id") or title,
                "title": title,
            }
        )
    return {
        "statutes": _dedup_list(statutes, key="statute_id"),
        "precedents": _dedup_list(precedents_summary, key="precedent_id"),
    }


def _build_precedent_graph(
    chain_steps: list[dict],
    arguments: list[dict],
    precedents: list[dict],
) -> tuple[list[dict], list[dict]]:
    if not precedents:
        return [], []

    summary = []
    for p in precedents:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        summary.append(
            {
                "precedent_id": p.get("precedent_id") or title,
                "title": title,
                "summary": p.get("summary") or "",
                "excerpt": p.get("excerpt") or "",
                "materia": p.get("materia") or "",
                "url": p.get("url") or "",
                "score": p.get("score"),
            }
        )
    summary = _dedup_list(summary, key="precedent_id")
    nodes = []
    id_map = {}
    for idx, p in enumerate(summary, start=1):
        node_id = f"PR{idx}"
        nodes.append(
            {
                "id": node_id,
                "precedent_id": p.get("precedent_id"),
                "title": p.get("title"),
                "summary": p.get("summary"),
                "excerpt": p.get("excerpt"),
                "materia": p.get("materia"),
                "url": p.get("url"),
                "score": p.get("score"),
            }
        )
        id_map[p.get("precedent_id")] = node_id

    links: list[dict] = []
    seen_links = set()

    def add_links(citations: dict, target_id: str) -> None:
        for prec in citations.get("precedents", []):
            prec_id = prec.get("precedent_id") or prec.get("title")
            node_id = id_map.get(prec_id)
            if not node_id:
                continue
            key = (node_id, target_id)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append({"from": node_id, "to": target_id, "type": "supports"})

    for step in chain_steps:
        add_links(step.get("citations", {}), step.get("id", ""))

    for arg in arguments:
        for premise in arg.get("premises", []):
            add_links(premise.get("citations", {}), premise.get("id", ""))
        conclusion = arg.get("conclusion") or {}
        if conclusion:
            add_links(conclusion.get("citations", {}), conclusion.get("id", ""))

    return nodes, links


def _dedup_list(items: list[dict], key: str) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        value = item.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(item)
    return result
