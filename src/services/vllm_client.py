"""Generic vLLM offline inference client.

Drop-in replacement for any LangChain chat model. Models are loaded once
into a global registry via load_models(); ChatVLLMOffline instances look
them up by alias at call time.

Configuration (call before load_models()):
    set_alias_map({"my_alias": "org/model-id", ...})
    set_reasoning_aliases({"my_alias"})   # aliases that emit <think>…</think>

Optional GPU/hardware overrides via environment variables (vLLM defaults apply if unset):
    VLLM_TENSOR_PARALLEL_SIZE
    VLLM_GPU_MEMORY_UTILIZATION
    VLLM_HF_CACHE_DIR
    VLLM_QUANTIZATION
    VLLM_MAX_MODEL_LEN
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    from vllm import LLM, SamplingParams  # type: ignore[import]

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Project-specific configuration (set via set_alias_map / set_reasoning_aliases)
# ──────────────────────────────────────────────────────────────────────────────

_alias_map: dict[str, str] = {}
_reasoning_aliases: frozenset[str] = frozenset()
_config_lock = threading.Lock()


def set_alias_map(mapping: dict[str, str]) -> None:
    """Register alias → HuggingFace model ID mapping.

    Call before load_models(). If an alias is not in the map, it is used
    as-is (i.e. treated as a direct HuggingFace model ID).
    """
    global _alias_map
    with _config_lock:
        _alias_map = dict(mapping)


def set_reasoning_aliases(aliases: set[str] | frozenset[str]) -> None:
    """Register aliases whose output contains <think>…</think> tokens to strip."""
    global _reasoning_aliases
    with _config_lock:
        _reasoning_aliases = frozenset(aliases)


# ──────────────────────────────────────────────────────────────────────────────
# Global model registry
# ──────────────────────────────────────────────────────────────────────────────

_registry: dict[str, Any] = {}
_registry_lock = threading.Lock()
_default_alias: Optional[str] = None


def set_default_model(alias: str) -> None:
    """Pin the fallback model for aliases not present in the registry."""
    global _default_alias
    _default_alias = alias


def load_models(aliases: list[str]) -> None:
    """Load vLLM LLM instances into the global registry.

    Hardware parameters are resolved from environment variables (see module
    docstring). Already-loaded aliases are skipped.
    """
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vLLM is not installed. Run: pip install vllm")

    with _config_lock:
        current_map = dict(_alias_map)

    with _registry_lock:
        for alias in aliases:
            if alias in _registry:
                continue
            hf_model = current_map.get(alias, alias)
            kwargs: dict[str, Any] = {"model": hf_model, "trust_remote_code": True}
            _tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE")
            if _tp:
                kwargs["tensor_parallel_size"] = int(_tp)
            _gmu = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
            if _gmu:
                kwargs["gpu_memory_utilization"] = float(_gmu)
            _cache = os.environ.get("VLLM_HF_CACHE_DIR")
            if _cache:
                kwargs["download_dir"] = _cache
            _quant = os.environ.get("VLLM_QUANTIZATION")
            if _quant:
                kwargs["quantization"] = _quant
            _mml = os.environ.get("VLLM_MAX_MODEL_LEN")
            if _mml:
                kwargs["max_model_len"] = int(_mml)

            print(f"[vLLM] Loading {alias} → {hf_model} …")
            _registry[alias] = LLM(**kwargs)  # type: ignore[misc]
            print(f"[vLLM] {alias} ready ✓")


def get_loaded_aliases() -> list[str]:
    with _registry_lock:
        return list(_registry.keys())


def unload_model(alias: str) -> None:
    """Remove a model from the registry and free GPU memory."""
    global _default_alias
    with _registry_lock:
        if alias not in _registry:
            return
        del _registry[alias]
        if _default_alias == alias:
            _default_alias = None

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _get_llm(alias: str) -> Any:
    """Return the LLM for *alias* with fallback to default or first loaded."""
    with _registry_lock:
        if alias in _registry:
            return _registry[alias]
        target = _default_alias or (next(iter(_registry)) if _registry else None)
        if target and target in _registry:
            return _registry[target]
    raise RuntimeError(
        f"No vLLM model loaded for alias '{alias}'. "
        "Call load_models() before running the pipeline."
    )


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


def _messages_to_prompt(messages: Sequence[BaseMessage], llm: Any) -> str:
    """Convert LangChain messages to a string using the tokenizer's chat template."""
    raw = [{"role": _lc_role(m), "content": str(m.content)} for m in messages]
    try:
        tok = llm.get_tokenizer()
        return tok.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = [f"<|{r['role']}|>\n{r['content']}" for r in raw]
        parts.append("<|assistant|>")
        return "\n".join(parts)


def _strip_thinking(text: str) -> str:
    """Strip <think>…</think> reasoning blocks (e.g. DeepSeek R1 output)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ──────────────────────────────────────────────────────────────────────────────
# LangChain-compatible chat model
# ──────────────────────────────────────────────────────────────────────────────


class ChatVLLMOffline(BaseChatModel):
    """LangChain BaseChatModel backed by vLLM offline (in-process) inference.

    Stores only the model alias; the LLM instance is looked up from the
    global registry at call time. Call load_models() before first use.
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: Optional[int] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm = _get_llm(self.model)
        prompt = _messages_to_prompt(messages, llm)

        sampling = SamplingParams(  # type: ignore[misc]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop or [],
            seed=self.seed,
        )
        outputs = llm.generate([prompt], sampling)
        text: str = outputs[0].outputs[0].text

        with _config_lock:
            is_reasoning = self.model in _reasoning_aliases
        if is_reasoning:
            text = _strip_thinking(text)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        result = self._generate(messages, stop=stop, **kwargs)
        yield result.generations[0]

    @property
    def _llm_type(self) -> str:
        return "vllm-offline"
