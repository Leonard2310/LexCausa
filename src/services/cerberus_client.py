"""Cerberus (llama.cpp) inference client for LexCausa.

Drop-in replacement for the old vLLM offline client. Models are **served
externally** by Cerberus (`cerberus up`, llama.cpp backend on IBiSCo) and reached
over an OpenAI-compatible HTTP API; the endpoint map lives in ``endpoints.json``.
This module exposes:

* :class:`ChatCerberus` — a LangChain ``BaseChatModel`` that routes every call to
  a Cerberus-served model, picked by its **label**. It is a drop-in for the former
  ``ChatVLLMOffline`` (same ``model`` / ``temperature`` / ``max_tokens`` / ``seed``
  fields), so the reasoning pipeline is unchanged.
* module-level helpers mirroring the old vLLM registry API
  (``set_alias_map`` / ``set_reasoning_aliases`` / ``set_default_model`` /
  ``load_models`` / ``unload_model``) so the experiment entry point keeps working.

Because Cerberus serves the models out-of-process, ``load_models`` /
``unload_model`` are **not** GPU load/unload cycles: they only validate/announce
availability against ``endpoints.json``. Bringing models up/down is done by
``cerberus up`` (see Cerberus/docs/progetto.md).

Endpoint map resolution (via cerberus.CerberusClient):
    1. CERBERUS_ENDPOINTS env var, or path set with set_endpoints_path();
    2. project_dir/endpoints.json (set_project_dir() or CERBERUS_PROJECT_DIR);
    3. ./endpoints.json in the current working directory.

Off-cluster safe: importing this module and building the client never touch the
filesystem or network, so the pipeline can be imported/tested on a laptop.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    from cerberus import CerberusClient  # type: ignore[import]

    _CERBERUS_AVAILABLE = True
except ImportError:  # off-cluster / cerberus package not installed
    CerberusClient = None  # type: ignore[assignment,misc]
    _CERBERUS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Project-specific configuration (set via set_alias_map / set_reasoning_aliases)
# ──────────────────────────────────────────────────────────────────────────────

_alias_map: dict[str, str] = {}
_reasoning_aliases: frozenset[str] = frozenset()
_reasoning_effort: dict[str, str] = {}
_default_alias: Optional[str] = None
_config_lock = threading.Lock()


def set_alias_map(mapping: dict[str, str]) -> None:
    """Register LexCausa alias → Cerberus model *label* mapping.

    Call before running the pipeline. If an alias is not in the map, it is used
    as-is (i.e. treated as a Cerberus label directly). This mirrors the old vLLM
    client, except the target is a served label (from ``models.conf``) rather than
    a HuggingFace model id.
    """
    global _alias_map
    with _config_lock:
        _alias_map = dict(mapping)


def set_reasoning_aliases(aliases: set[str] | frozenset[str]) -> None:
    """Register aliases whose model emits chain-of-thought reasoning.

    Kept for API compatibility with the old client. With Cerberus the reasoning
    trace is separated server-side (``reasoning_content``) and stripped by the
    client, so ``ChatCerberus`` already returns a clean answer regardless. The
    set is used only to nudge the per-request ``reasoning`` flag on/off; the real
    default is controlled by ``reasoning`` in ``models.conf``.
    """
    global _reasoning_aliases
    with _config_lock:
        _reasoning_aliases = frozenset(aliases)


def set_reasoning_effort(mapping: dict[str, str]) -> None:
    """Register per-alias reasoning effort (e.g. {"gpt_oss_120b": "low"}).

    Applied as ``chat_template_kwargs.reasoning_effort`` on the request — used by
    harmony models (gpt-oss) to bound the thinking budget. Aliases not in the map
    use the model/server default.
    """
    global _reasoning_effort
    with _config_lock:
        _reasoning_effort = dict(mapping)


def set_default_model(alias: str) -> None:
    """Pin the fallback label for aliases not present in the endpoint map."""
    global _default_alias
    _default_alias = alias


_configured = False


def ensure_configured(
    alias_map: dict[str, str],
    reasoning_aliases: "set[str] | frozenset[str]",
    reasoning_effort: dict[str, str],
) -> None:
    """Apply alias_map/reasoning_aliases/reasoning_effort once per process.

    Only ``scripts/run_multi_doe_ibisco.py`` calls set_alias_map() explicitly at
    startup. Other entry points (``get_chat_groq()`` on the API/pipeline paths,
    e.g. PolisherEvaluator's attack-type classification) build a ChatCerberus
    without ever configuring the maps, so every alias would silently fall back to
    the fixed model. Idempotent: no-op after the first call.
    """
    global _configured
    with _config_lock:
        if _configured:
            return
        _configured = True
    set_alias_map(alias_map)
    set_reasoning_aliases(reasoning_aliases)
    set_reasoning_effort(reasoning_effort)


def _resolve_label(alias: str, served: Optional[set[str]] = None) -> str:
    """Map a LexCausa alias to its Cerberus served label.

    Mirrors the old vLLM registry semantics: an alias maps through ``_alias_map``
    (identity if absent); if the mapped label is **not served** and a default
    (``fixed-model``) is pinned, fall back to the default's label. This is what
    lets support phases (retrieval / classifier / AQA) — which ask for provider
    ids that are not Cerberus labels — run on the pinned fixed model. ``served``
    is the set of labels currently in endpoints.json; when unknown, no fallback
    is applied and the mapped label is returned as-is.
    """
    with _config_lock:
        mapped = _alias_map.get(alias, alias)
        default_label = (
            _alias_map.get(_default_alias, _default_alias) if _default_alias else None
        )
    if served is None or mapped in served:
        return mapped
    if default_label and default_label in served:
        return default_label
    return mapped  # not served and no usable default → let chat() fail clearly


# ──────────────────────────────────────────────────────────────────────────────
# Shared Cerberus client (lazy, thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

_client: Any = None
_client_lock = threading.Lock()
_endpoints_path: Optional[str] = None
_project_dir: Optional[str] = None


def set_endpoints_path(path: Optional[str]) -> None:
    """Point the client at an explicit endpoints.json (overrides env / cwd)."""
    global _endpoints_path, _client, _served_cache
    with _client_lock:
        _endpoints_path = path
        _client = None  # force rebuild with the new location
        _served_cache = None


def set_project_dir(path: Optional[str]) -> None:
    """Point the client at a project directory containing endpoints.json."""
    global _project_dir, _client, _served_cache
    with _client_lock:
        _project_dir = path
        _client = None
        _served_cache = None


def get_client() -> Any:
    """Return the shared CerberusClient, building it lazily.

    Raises RuntimeError if the cerberus package is not installed (e.g. off-cluster
    without `pip install -e Cerberus`).
    """
    global _client
    if not _CERBERUS_AVAILABLE:
        raise RuntimeError(
            "The 'cerberus' package is not installed. On IBiSCo run "
            "`pip install -e /path/to/Cerberus` in the active env so "
            "`from cerberus import CerberusClient` works."
        )
    with _client_lock:
        if _client is None:
            env_project = os.environ.get("CERBERUS_PROJECT_DIR")
            _client = CerberusClient(
                endpoints_path=_endpoints_path,
                project_dir=_project_dir or env_project,
            )
        return _client


_served_cache: Optional[set[str]] = None
# Runtime visibility into label resolution. CERBERUS_DEBUG=1 logs the served-set
# read and every resolution; a FALLBACK (a model's label not served → fixed model)
# is ALWAYS warned once — that silently runs a cell on the WRONG model.
_DEBUG = bool(os.environ.get("CERBERUS_DEBUG"))
_warned_resolutions: set = set()


def _served_labels() -> Optional[set[str]]:
    """Set of labels currently served (cached); None if the map is unavailable.

    The endpoint map is stable for the lifetime of a `cerberus up`, so it is read
    once and cached. set_endpoints_path()/set_project_dir() reset the client and
    this cache is refreshed lazily on the next call.
    """
    global _served_cache
    if _served_cache is None:
        try:
            client = get_client()
            if client.is_available():
                _served_cache = set(client.list_models())
                if _DEBUG:
                    print(
                        f"[cerberus] served labels: {sorted(_served_cache)}", flush=True
                    )
        except Exception:
            return None
    return _served_cache


def support_label() -> str:
    """Served label to use for aux/support calls (classifier, taxonomy, retrieval
    filter, etc.).

    These phases are not DoE factors: they run on the pinned fixed model
    (``set_default_model``). Falls back to the first served label if no default
    was pinned. Empty string if nothing is served (caller will fail clearly).
    """
    with _config_lock:
        default = _default_alias
        mapped_default = _alias_map.get(default, default) if default else None
    if mapped_default:
        return mapped_default
    served = _served_labels() or set()
    return next(iter(sorted(served)), "")


# ──────────────────────────────────────────────────────────────────────────────
# Registry-compatible helpers (no in-process GPU load — Cerberus serves models)
# ──────────────────────────────────────────────────────────────────────────────


def load_models(aliases: list[str]) -> None:
    """Validate that the requested aliases are served by Cerberus.

    Unlike the vLLM client this does **not** load weights into GPU memory — the
    models are already served by `cerberus up`. It resolves each alias to its
    label and checks it against ``endpoints.json``, printing a clear warning for
    any label that is not currently up (without raising, so a not-yet-started
    server does not abort orchestration setup).
    """
    client = get_client()
    if not client.is_available():
        print(
            "[cerberus] endpoints.json not found — run 'cerberus up' first "
            "(or set CERBERUS_ENDPOINTS). Models will fail at call time."
        )
        return
    served = set(client.list_models())
    for alias in aliases:
        label = _resolve_label(alias)
        if label in served:
            print(f"[cerberus] {alias} → '{label}' ready ✓")
        else:
            print(
                f"[cerberus] WARNING: {alias} → '{label}' not served "
                f"(available: {sorted(served)})"
            )


def unload_model(alias: str) -> None:
    """No-op: Cerberus owns the model lifecycle (stop with 'cerberus down'/Ctrl-C).

    Kept so the experiment orchestrator's load/unload calls stay valid without
    special-casing the backend.
    """
    global _default_alias
    if _default_alias == alias:
        _default_alias = None


def get_loaded_aliases() -> list[str]:
    """Return the labels currently served by Cerberus (best-effort)."""
    try:
        client = get_client()
        if client.is_available():
            return client.list_models()
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────


def _lc_role(msg: BaseMessage) -> str:
    from langchain_core.messages import AIMessage as AI
    from langchain_core.messages import SystemMessage as SM

    if isinstance(msg, SM):
        return "system"
    if isinstance(msg, AI):
        return "assistant"
    return "user"


def _messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI-style {role, content} dicts.

    The llama.cpp server applies the model's chat template itself, so we hand it
    structured messages rather than a pre-rendered prompt string.
    """
    return [{"role": _lc_role(m), "content": str(m.content)} for m in messages]


# ──────────────────────────────────────────────────────────────────────────────
# LangChain-compatible chat model
# ──────────────────────────────────────────────────────────────────────────────


class ChatCerberus(BaseChatModel):
    """LangChain BaseChatModel backed by a Cerberus-served llama.cpp model.

    Stores only the model alias/label; the served endpoint is resolved from
    ``endpoints.json`` at call time via the shared :class:`CerberusClient`. The
    reasoning trace (thinking) is separated server-side and stripped by the
    client, so ``content`` is always the clean final answer.
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: Optional[int] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager  # required by the LangChain _generate override, unused here
        client = get_client()
        served = _served_labels()
        label = _resolve_label(self.model, served)
        # Runtime visibility: warn ONCE if this alias fell back to the fixed model
        # (its own label isn't served) — that silently runs the cell on the wrong
        # model. With CERBERUS_DEBUG=1, log every resolution.
        with _config_lock:
            _mapped = _alias_map.get(self.model, self.model)
        if (_DEBUG or label != _mapped) and (
            self.model,
            label,
        ) not in _warned_resolutions:
            _warned_resolutions.add((self.model, label))
            _tag = "⚠️ FALLBACK" if label != _mapped else "resolve"
            print(
                f"[cerberus] {_tag}: model={self.model!r} mapped={_mapped!r} -> "
                f"label={label!r} | served={sorted(served) if served else served}",
                flush=True,
            )
        payload = _messages_to_openai(messages)

        # Reasoning applies to the *served label*: an alias that fell back to the
        # fixed support model must not force thinking meant for a reasoning model.
        with _config_lock:
            reasoning_labels = {_alias_map.get(a, a) for a in _reasoning_aliases}
            effort_by_label = {
                _alias_map.get(a, a): e for a, e in _reasoning_effort.items()
            }
        is_reasoning = label in reasoning_labels
        effort = effort_by_label.get(label)

        # reasoning=None leaves the server default (models.conf). For aliases we
        # know are reasoning models, request thinking explicitly (best-effort);
        # the client still returns a clean, think-free `content` either way.
        reasoning_flag: Optional[bool] = True if is_reasoning else None

        extra: dict[str, Any] = {}
        if stop:
            extra["stop"] = stop
        if self.seed is not None:
            extra["seed"] = self.seed
        # Forward JSON mode so JSON-producing calls (planners, gates, target maps)
        # are grammar-constrained by llama.cpp — same as the cloud backends. Without
        # it the model emits prose/fences and json.loads fails ("char 0"). If the
        # server rejects response_format, callers already retry without it.
        response_format = kwargs.get("response_format")
        if response_format is not None:
            extra["response_format"] = response_format
        # Per-model reasoning effort (e.g. gpt-oss "low") → harmony chat template.
        # CerberusClient merges enable_thinking into the same chat_template_kwargs.
        if effort:
            extra.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})[
                "reasoning_effort"
            ] = effort

        resp = client.chat(
            label,
            payload,
            reasoning=reasoning_flag,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **extra,
        )
        # Espone la usage del server (resp.raw.usage) e la registra in
        # _usage_stats: senza questo le chiamate Cerberus bypassano il conteggio
        # (resilient_chat_call salta _record_llm_stats sul fast-path non-Groq) e
        # le colonne token/chiamate di metrics.csv restano a ~0.
        usage = getattr(getattr(resp, "raw", None), "usage", None)
        usage_metadata = None
        if usage is not None:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
            usage_metadata = {
                "input_tokens": pt,
                "output_tokens": ct,
                "total_tokens": tt,
            }
            try:
                from services.usage_stats import get_usage_stats

                get_usage_stats().record_llm_call(
                    provider="cerberus",
                    model=label,
                    source="chat_cerberus_generate",
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                )
            except Exception:
                pass

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=resp.content or "",
                        usage_metadata=usage_metadata,
                    )
                )
            ]
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        # Mirror the old offline client: emit the full result as a single chunk.
        result = self._generate(messages, stop=stop, **kwargs)
        yield result.generations[0]

    @property
    def _llm_type(self) -> str:
        return "cerberus-llamacpp"
