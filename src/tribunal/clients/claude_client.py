"""Claude-family generation client.

"Thinking on/off" is not a single uniform API knob across Claude models —
Haiku 4.5 predates adaptive thinking and the `effort` parameter, while
Sonnet 5 / Opus 4.8 support both. Rather than hardcode per-model branches
that could silently drift out of date, this module queries the live Models
API for each model's actual capabilities the first time it's used, caches
the result, and picks the correct request shape. If a model can't honor the
requested thinking mode at all, that is recorded in the output record
rather than silently ignored — a mismatch here would quietly corrupt the
experiment rather than fail loudly.

`effort` (when the model supports it) is held constant across the on/off
thinking conditions, so thinking mode is the only variable that changes
between the two arms of the on/off comparison for a given model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from anthropic import Anthropic

from ..config import CLAUDE_MODELS, MAX_RESPONSE_TOKENS


@dataclass
class GenerationResult:
    model_key: str
    model_id: str
    thinking_requested: str  # "on" | "off"
    thinking_applied: bool  # did the request actually invoke thinking
    thinking_method: str  # "adaptive" | "enabled_budget" | "unsupported"
    effort_applied: str | None
    response_text: str
    thinking_text: str | None
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    raw_error: str | None = None


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or put it in a .env "
                "file (see .env.example), before running any Claude "
                "generation or judging step."
            )
        _client = Anthropic()
    return _client


def get_client() -> Anthropic:
    """Public accessor so other modules can reuse the same
    lazily-constructed, key-checked client instead of duplicating the
    ANTHROPIC_API_KEY check."""
    return _get_client()


@lru_cache(maxsize=None)
def _capabilities(model_id: str) -> dict[str, Any]:
    """Fetch and cache a model's capability tree from the Models API.

    The installed SDK (anthropic 0.120.x) returns `capabilities` as a typed
    pydantic object (ModelCapabilities), not a plain dict — `.model_dump()`
    normalizes it to nested dicts so the rest of this module can use
    uniform `.get(...)` access regardless of SDK version.
    """
    client = _get_client()
    model = client.models.retrieve(model_id)
    caps = model.capabilities
    return caps.model_dump() if hasattr(caps, "model_dump") else dict(caps)


def _thinking_request_shape(
    model_id: str, thinking_on: bool
) -> tuple[dict | None, str, bool]:
    """Return (thinking_param, thinking_method, thinking_applied) for this
    model + requested mode, based on live capability data rather than a
    hardcoded per-model-family assumption.
    """
    caps = _capabilities(model_id)
    types = caps.get("thinking", {}).get("types", {})
    adaptive_supported = types.get("adaptive", {}).get("supported", False)
    enabled_supported = types.get("enabled", {}).get("supported", False)

    if adaptive_supported:
        if thinking_on:
            return {"type": "adaptive", "display": "summarized"}, "adaptive", True
        disabled_supported = types.get("disabled", {}).get("supported", False)
        if disabled_supported:
            return {"type": "disabled"}, "adaptive", False
        # Some adaptive-only models (e.g. Claude Fable 5) reject an explicit
        # `disabled` and expect the param omitted entirely for the off case.
        return None, "adaptive", False

    if enabled_supported:
        if thinking_on:
            return {"type": "enabled", "budget_tokens": 8000}, "enabled_budget", True
        return None, "enabled_budget", False

    # No thinking support at all on this model — the "on" condition cannot
    # be honored. Recorded explicitly so downstream analysis can exclude or
    # flag these rows rather than silently treating them as a real contrast.
    return None, "unsupported", False


def _effort_for(model_id: str) -> str | None:
    caps = _capabilities(model_id)
    if caps.get("effort", {}).get("high", {}).get("supported", False):
        return "high"
    return None


def generate(
    model_key: str,
    prompt: str,
    thinking_on: bool,
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> GenerationResult:
    """Generate one response from a Claude-family model under one thinking
    condition. `model_key` is a key into config.CLAUDE_MODELS (e.g.
    "claude-opus"), not a raw model ID string.
    """
    model_id = CLAUDE_MODELS[model_key]["id"]
    client = _get_client()

    thinking_param, thinking_method, thinking_applied = _thinking_request_shape(
        model_id, thinking_on
    )
    effort = _effort_for(model_id)

    # Pre-adaptive models (`enabled_budget` method) require max_tokens to
    # strictly exceed thinking.budget_tokens — the response itself needs
    # headroom on top of the thinking budget, or the API 400s.
    effective_max_tokens = max_tokens
    if thinking_param is not None and thinking_param.get("type") == "enabled":
        budget = thinking_param["budget_tokens"]
        effective_max_tokens = max(max_tokens, budget + 2048)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": effective_max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_param is not None:
        kwargs["thinking"] = thinking_param
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}

    start = time.monotonic()
    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim in the record
        return GenerationResult(
            model_key=model_key,
            model_id=model_id,
            thinking_requested="on" if thinking_on else "off",
            thinking_applied=thinking_applied,
            thinking_method=thinking_method,
            effort_applied=effort,
            response_text="",
            thinking_text=None,
            stop_reason="error",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=time.monotonic() - start,
            raw_error=str(exc),
        )
    latency = time.monotonic() - start

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking or "")

    return GenerationResult(
        model_key=model_key,
        model_id=model_id,
        thinking_requested="on" if thinking_on else "off",
        thinking_applied=thinking_applied,
        thinking_method=thinking_method,
        effort_applied=effort,
        response_text="\n".join(text_parts).strip(),
        thinking_text="\n".join(thinking_parts).strip() or None,
        stop_reason=response.stop_reason or "",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_seconds=latency,
    )
