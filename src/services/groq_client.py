"""
Resilient Groq Client for LexCausa.

Provides automatic retry, API key rotation, and model fallback for both:
- groq.Groq (raw SDK, used by ClaimClassifier)
- langchain_groq.ChatGroq (LangChain wrapper, used by agents and StanceClassifier)

Error-handling strategy:
- Model over-capacity (503 "over capacity") → switch to fallback MODEL immediately
- Rate limit (429 "too many requests")      → rotate API KEY, keep same model
- Transient errors (500, 502, 504, timeout) → retry with backoff, then rotate key
"""

import logging
import threading
import time
from typing import Callable, Optional

from groq import Groq
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger("lexcausa.groq_client")


class AllKeysRateLimitedError(Exception):
    """Raised when every available API key is rate-limited."""

    def __init__(self, n_keys: int, model: str):
        self.n_keys = n_keys
        self.model = model
        super().__init__(
            f"All {n_keys} Groq API keys are rate-limited for model '{model}'. "
            f"Wait a few minutes or add more GROQ_API_KEY_V* entries to .env."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state (shared across all callers in the same process)
# ─────────────────────────────────────────────────────────────────────────────
_key_lock = threading.Lock()
_current_key_index: int = 0

# Model-down cache: remembers which models are down so we don't retry them
# on every single call.  Maps model_name → timestamp when it was marked down.
_MODEL_DOWN_TTL: float = settings.model_down_ttl
_model_down_cache: dict[str, float] = {}
_model_lock = threading.Lock()


def _mark_model_down(model: str) -> None:
    """Record that *model* is currently unavailable."""
    with _model_lock:
        _model_down_cache[model] = time.time()
        logger.warning(
            "📛 [groq_client] Model %s marked as DOWN for %ds",
            model,
            int(_MODEL_DOWN_TTL),
        )


def _is_model_cached_down(model: str) -> bool:
    """Return True if *model* was recently marked as down (within TTL)."""
    with _model_lock:
        ts = _model_down_cache.get(model)
        if ts is None:
            return False
        if time.time() - ts > _MODEL_DOWN_TTL:
            # TTL expired → give the model another chance
            del _model_down_cache[model]
            logger.info("🔄 [groq_client] Model %s TTL expired, will retry it", model)
            return False
        return True


def _rotate_key() -> tuple[str, int]:
    """
    Rotate to the next available API key (thread-safe).

    Returns:
        (new_api_key, new_index)
    """
    global _current_key_index
    keys = settings.groq_api_keys
    if not keys:
        raise ValueError(
            "No Groq API keys configured. Set GROQ_API_KEY_V1 (and optionally V2, V3, …) in .env"
        )
    with _key_lock:
        _current_key_index = (_current_key_index + 1) % len(keys)
        return keys[_current_key_index], _current_key_index


def _current_key() -> str:
    """Get the current API key."""
    keys = settings.groq_api_keys
    if not keys:
        raise ValueError(
            "No Groq API keys configured. Set GROQ_API_KEY_V1 (and optionally V2, V3, …) in .env"
        )
    with _key_lock:
        idx = _current_key_index % len(keys)
        return keys[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────────────────────────────────────


def _is_model_down(exc: Exception) -> bool:
    """True when the error is about the MODEL being unavailable / over capacity."""
    exc_str = str(exc).lower()
    return any(
        marker in exc_str
        for marker in ("over capacity", "currently unavailable", "model not available")
    )


def _is_rate_limit(exc: Exception) -> bool:
    """True when the error is a per-key rate-limit (429)."""
    exc_str = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return any(
        marker in exc_str
        for marker in ("rate limit", "rate_limit", "too many requests")
    )


def _is_retryable(exc: Exception) -> bool:
    """True for any transient/retryable error (includes model-down and rate-limit)."""
    if _is_model_down(exc) or _is_rate_limit(exc):
        return True
    exc_str = str(exc).lower()
    retryable_codes = ("503", "429", "500", "502", "504")
    retryable_messages = (
        "timeout",
        "timed out",
        "connection error",
        "service unavailable",
        "internal_server_error",
    )
    for code in retryable_codes:
        if code in exc_str:
            return True
    for msg in retryable_messages:
        if msg in exc_str:
            return True
    status = getattr(exc, "status_code", None)
    if status and status in (429, 500, 502, 503, 504):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Core resilient loop (shared logic for all three wrappers)
# ─────────────────────────────────────────────────────────────────────────────


def _resilient_loop(
    execute_fn,
    *,
    max_retries: Optional[int] = None,
    label: str = "Groq",
    model_order: Optional[list[str]] = None,
):
    """
    Generic resilient execution loop.

    Args:
        execute_fn: ``(key: str, model: str) -> result``
                    Called for each attempt.
        max_retries: Per-combination retry cap.
        label: Label for log messages.

    Strategy:
    - On *model-down* → skip remaining retries/keys for this model, jump to
      fallback model immediately.
    - On *rate-limit* → rotate API key, keep the same model.  If **every**
      key is rate-limited, raise ``AllKeysRateLimitedError`` immediately
      instead of looping forever.
    - On other transient errors → retry with exponential backoff, then rotate key.
    """
    retries = max_retries or settings.groq_max_retries
    # Allow per-call model order overrides (used by Reasoner/CounterReasoner).
    if model_order:
        models: list[str] = []
        for m in model_order:
            m_norm = str(m or "").strip()
            if m_norm and m_norm not in models:
                models.append(m_norm)
    else:
        models = []
    if not models:
        models = list(settings.groq_models)  # copy so we can iterate safely
    keys = settings.groq_api_keys
    n_keys = len(keys)
    base_delay = settings.groq_retry_base_delay

    if not keys:
        raise ValueError(
            "No Groq API keys configured. "
            "Set GROQ_API_KEY_V1 (and optionally V2, V3, …) in .env"
        )

    logger.info(
        "🔑 [%s] %d API key(s) available, %d model(s) configured",
        label,
        n_keys,
        len(models),
    )

    last_exc: Optional[Exception] = None
    model_idx = 0

    # ── Skip models already known to be down ─────────────────────────────
    while model_idx < len(models) and _is_model_cached_down(models[model_idx]):
        logger.info(
            "⏩ [%s] Skipping model %s (cached as down)",
            label,
            models[model_idx],
        )
        model_idx += 1

    if model_idx >= len(models):
        # Every model is cached-down – reset and try anyway
        logger.warning(
            "⚠️ [%s] All models cached as down, retrying from first model",
            label,
        )
        model_idx = 0

    while model_idx < len(models):
        model = models[model_idx]
        # Track which keys have been rate-limited for THIS model
        rate_limited_keys: set[int] = set()
        keys_tried = 0

        while keys_tried < n_keys:
            key = _current_key()
            cur_idx = _current_key_index

            for attempt in range(1, retries + 1):
                try:
                    return execute_fn(key, model)
                except Exception as exc:
                    last_exc = exc

                    if not _is_retryable(exc):
                        raise  # Non-retryable (e.g. 400 Bad Request) → fail fast

                    # ── Model down → switch model immediately ────────────
                    if _is_model_down(exc):
                        _mark_model_down(model)
                        logger.warning(
                            "🔀 [%s] Model %s is down, switching to fallback "
                            "immediately: %s",
                            label,
                            model,
                            exc,
                        )
                        model_idx += 1
                        if model_idx < len(models):
                            logger.warning(
                                "🔀 [%s] Fallback model: %s",
                                label,
                                models[model_idx],
                            )
                        # break out of both the attempt and keys loops
                        keys_tried = n_keys  # exit keys loop after break
                        break  # exit attempt loop

                    # ── Rate limit → rotate key immediately ──────────────
                    if _is_rate_limit(exc):
                        rate_limited_keys.add(cur_idx)
                        logger.warning(
                            "🔄 [%s] Rate limit hit (key %d/%d), "
                            "rotating API key: %s",
                            label,
                            cur_idx + 1,
                            n_keys,
                            exc,
                        )
                        # Check if ALL keys are now rate-limited
                        if len(rate_limited_keys) >= n_keys:
                            logger.error(
                                "🚫 [%s] All %d API keys are rate-limited "
                                "for model '%s'. Stopping.",
                                label,
                                n_keys,
                                model,
                            )
                            raise AllKeysRateLimitedError(n_keys, model) from exc
                        _rotate_key()
                        keys_tried += 1
                        break  # exit attempt loop, go to next key

                    # ── Other transient error → backoff + retry ──────────
                    logger.warning(
                        "⚠️ [%s] Transient error (model=%s, key %d/%d, "
                        "attempt=%d/%d): %s",
                        label,
                        model,
                        cur_idx + 1,
                        n_keys,
                        attempt,
                        retries,
                        exc,
                    )
                    if attempt < retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        time.sleep(delay)
            else:
                # All retry attempts exhausted for this key (no break) → rotate key
                logger.warning(
                    "🔄 [%s] Retries exhausted for key %d/%d, rotating key",
                    label,
                    _current_key_index + 1,
                    n_keys,
                )
                _rotate_key()
                keys_tried += 1
                continue

            # We broke out of the attempt loop (rate-limit or model-down)
            continue

        # If we got here for model-down, model_idx was already incremented above
        if model_idx >= len(models):
            break
        # If all keys exhausted normally, move to next model
        if keys_tried >= n_keys and model == models[model_idx]:
            logger.warning(
                "🔀 [%s] All %d keys exhausted for model %s, trying fallback",
                label,
                n_keys,
                model,
            )
            model_idx += 1

    # Everything exhausted
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Resilient Groq SDK client (for ClaimClassifier)
# ─────────────────────────────────────────────────────────────────────────────


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """
    Create a Groq SDK client with the current (or specified) API key.

    Args:
        api_key: Explicit key override; if None, uses the current rotated key.

    Returns:
        groq.Groq instance.
    """
    key = api_key or _current_key()
    return Groq(api_key=key)


def resilient_groq_call(
    call_fn,
    *,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
):
    """
    Execute a Groq SDK call with retry + key rotation + model fallback.

    Args:
        call_fn: ``(client: Groq, model: str) -> T``
        max_retries: Override for settings.groq_max_retries.

    Returns:
        The result of call_fn on success.
    """

    def _execute(key: str, model: str):
        client = Groq(api_key=key)
        return call_fn(client, model)

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="Groq",
        model_order=model_order,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resilient ChatGroq (LangChain) factory
# ─────────────────────────────────────────────────────────────────────────────


def get_chat_groq(
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> ChatGroq:
    """
    Create a ChatGroq instance with the current API key.

    Args:
        model: Model name; defaults to first runtime model in settings.groq_models.
        temperature: LLM temperature; defaults to settings.llm_temperature.
        max_tokens: Max tokens; defaults to settings.llm_max_tokens.
        api_key: Explicit key override.

    Returns:
        ChatGroq instance.
    """
    key = api_key or _current_key()
    return ChatGroq(
        api_key=key,
        model=model or settings.groq_models[0],
        temperature=(
            temperature if temperature is not None else settings.llm_temperature
        ),
        max_tokens=max_tokens or settings.llm_max_tokens,
        **kwargs,
    )


def resilient_chat_call(
    llm_or_factory,
    messages,
    *,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
    **invoke_kwargs,
):
    """
    Execute a LangChain ChatGroq .invoke() with smart retry strategy.

    - Model over-capacity → switch model immediately
    - Rate limit          → rotate API key
    - Transient error     → backoff + retry
    """
    ref_llm = llm_or_factory  # used to read temperature / max_tokens

    def _execute(key: str, model: str):
        if callable(ref_llm) and not isinstance(ref_llm, ChatGroq):
            llm = ref_llm(key, model)
        else:
            llm = ChatGroq(
                api_key=key,
                model=model,
                temperature=getattr(ref_llm, "temperature", settings.llm_temperature),
                max_tokens=getattr(ref_llm, "max_tokens", settings.llm_max_tokens),
            )
        return llm.invoke(messages, **invoke_kwargs)

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="ChatGroq",
        model_order=model_order,
    )


def _chunk_to_text(chunk) -> str:
    """Extract text payload from a LangChain stream chunk."""
    if chunk is None:
        return ""

    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
                if txt:
                    parts.append(str(txt))
        return "".join(parts)
    return str(content)


def resilient_chat_stream(
    llm_or_factory,
    messages,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
    **stream_kwargs,
):
    """
    Execute a LangChain ChatGroq .stream() with retry/key-rotation/fallback.

    Emits token chunks via ``on_token`` and returns an AIMessage with full text.
    """
    ref_llm = llm_or_factory  # used to read temperature / max_tokens

    def _execute(key: str, model: str):
        if callable(ref_llm) and not isinstance(ref_llm, ChatGroq):
            llm = ref_llm(key, model)
        else:
            llm = ChatGroq(
                api_key=key,
                model=model,
                temperature=getattr(ref_llm, "temperature", settings.llm_temperature),
                max_tokens=getattr(ref_llm, "max_tokens", settings.llm_max_tokens),
            )

        pieces: list[str] = []
        for chunk in llm.stream(messages, **stream_kwargs):
            text = _chunk_to_text(chunk)
            if not text:
                continue
            pieces.append(text)
            if on_token is not None:
                on_token(text)
        return AIMessage(content="".join(pieces))

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="ChatGroqStream",
        model_order=model_order,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resilient LangGraph ReAct agent invocation
# ─────────────────────────────────────────────────────────────────────────────


def resilient_react_invoke(
    agent_builder,
    input_data: dict,
    *,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
):
    """
    Invoke a LangGraph ReAct agent with smart retry strategy.

    Args:
        agent_builder: ``(api_key: str, model: str) -> agent``
        input_data: The dict to pass to agent.invoke().
        max_retries: Override for settings.groq_max_retries.
    """

    def _execute(key: str, model: str):
        agent = agent_builder(key, model)
        return agent.invoke(input_data)

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="ReAct",
        model_order=model_order,
    )
