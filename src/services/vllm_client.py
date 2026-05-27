"""vLLM offline inference client for LexCausa (Ibisco/HPC mode).

Drop-in replacement for ChatGroq. Models are loaded once into a global
registry via load_models(); ChatVLLMOffline instances look them up by alias.

GPU/hardware parameters are read from environment variables so the HPC
operator can configure them without touching this code:
    VLLM_TENSOR_PARALLEL_SIZE   (default: 1)
    VLLM_GPU_MEMORY_UTILIZATION (default: 0.90)
    VLLM_HF_CACHE_DIR           (default: not set)
    VLLM_QUANTIZATION           (default: not set)
    VLLM_MAX_MODEL_LEN          (default: not set)
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
# Alias → HuggingFace model ID (mirrors Groq aliases used in DoE config)
# ──────────────────────────────────────────────────────────────────────────────

VLLM_ALIAS_MAP: dict[str, str] = {
    # Reasoning models (produce <think>…</think> tokens — stripped automatically)
    "deepseek_r1": "deepseek-ai/DeepSeek-R1",
    "gpt_oss_120b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # Non-reasoning models
    "qwen_25_72b": "Qwen/Qwen2.5-72B-Instruct",
    "groq_llama_3_3_70b_versatile": "meta-llama/Llama-3.3-70B-Instruct",
    "groq_llama_scout_17b": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
}

# Aliases whose output contains <think>…</think> reasoning tokens to strip
_REASONING_ALIASES: frozenset[str] = frozenset({"deepseek_r1", "gpt_oss_120b"})

# ──────────────────────────────────────────────────────────────────────────────
# Global model registry
# ──────────────────────────────────────────────────────────────────────────────

_registry: dict[str, Any] = {}  # alias -> vllm.LLM instance
_registry_lock = threading.Lock()
_default_alias: Optional[str] = None  # fallback for unknown aliases


def set_default_model(alias: str) -> None:
    """Set the model used for all pipeline phases except Reasoner/Counter.

    Call this after load_models() to pin the "utility" model (retrieval,
    filtering, AQA, consistency checker) independently from the DoE variable.
    When not set, unknown aliases fall back to the first model in the registry.
    """
    global _default_alias
    _default_alias = alias


def load_models(aliases: list[str]) -> None:
    """Load vLLM LLM instances into the global registry.

    Hardware parameters are resolved from environment variables (see module
    docstring). Call once at process startup; already-loaded aliases are skipped.
    """
    if not _VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not installed. On Ibisco: conda install vllm "
            "or pip install vllm"
        )
    tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = float(
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    )
    hf_cache_dir = os.environ.get("VLLM_HF_CACHE_DIR")
    quantization = os.environ.get("VLLM_QUANTIZATION")
    _max_len_str = os.environ.get("VLLM_MAX_MODEL_LEN")
    max_model_len = int(_max_len_str) if _max_len_str else None

    with _registry_lock:
        for alias in aliases:
            if alias in _registry:
                continue
            hf_model = VLLM_ALIAS_MAP.get(alias, alias)
            kwargs: dict[str, Any] = {
                "model": hf_model,
                "tensor_parallel_size": tensor_parallel_size,
                "trust_remote_code": True,
                "gpu_memory_utilization": gpu_memory_utilization,
            }
            if hf_cache_dir:
                kwargs["download_dir"] = hf_cache_dir
            if quantization:
                kwargs["quantization"] = quantization
            if max_model_len:
                kwargs["max_model_len"] = max_model_len

            print(f"[vLLM] Loading {alias} → {hf_model} …")
            _registry[alias] = LLM(**kwargs)  # type: ignore[misc]
            print(f"[vLLM] {alias} ready ✓")


def get_loaded_aliases() -> list[str]:
    with _registry_lock:
        return list(_registry.keys())


def unload_model(alias: str) -> None:
    """Remove a model from the registry and free GPU memory.

    Deletes the vLLM LLM object and calls torch.cuda.empty_cache() to
    ensure VRAM is reclaimed before loading the next model.
    """
    global _default_alias
    with _registry_lock:
        if alias not in _registry:
            return
        del _registry[alias]
        if _default_alias == alias:
            _default_alias = None

    # Explicit GC + CUDA cache flush so the next load_models() call has
    # the full VRAM budget available.
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _get_llm(alias: str) -> Any:
    """Return the LLM for *alias*.

    Resolution order:
    1. Exact alias match in registry
    2. Explicit default set via set_default_model()
    3. First model in registry (implicit fallback)

    This means all pipeline phases that use Groq aliases not present in the
    registry (retrieval filters, AQA, etc.) automatically route to the default/
    fixed model without any code changes to those components.
    """
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
    """Strip <think>…</think> reasoning blocks (DeepSeek R1 output)."""
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

        if any(k in self.model for k in _REASONING_ALIASES):
            text = _strip_thinking(text)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        # vLLM offline generates synchronously; yield the full result as one chunk.
        result = self._generate(messages, stop=stop, **kwargs)
        yield result.generations[0]

    @property
    def _llm_type(self) -> str:
        return "vllm-offline"
