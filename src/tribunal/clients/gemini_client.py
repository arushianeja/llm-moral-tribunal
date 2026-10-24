"""Gemini-family generation client.

Gated behind GEMINI_API_KEY (or GOOGLE_API_KEY, matching the google-genai
SDK's own resolution order). This module intentionally does NOT silently
skip Gemini models when no key is configured — every entry point here
raises a clear, actionable RuntimeError instead, so a partial run never
gets mistaken for a complete one.

Model IDs (see config.GEMINI_MODELS) and the thinking-control API surface
were confirmed via live web search in 2026, not from training-data recall
— Google iterates the Gemini API faster than most training cutoffs can
track. If a run 404s on a model ID, check
https://ai.google.dev/gemini-api/docs/latest-model and update
config.GEMINI_MODELS; do not guess a replacement ID.

IMPORTANT ASYMMETRY vs. the Claude side of this study, documented in the
README: Gemini 3-generation models expose thinking depth via
`thinking_level` ("minimal" | "low" | "medium" | "high"), and Google's own
docs describe "minimal" as approximating, not guaranteeing, a no-thinking
state — unlike Claude's Opus 4.8 / Sonnet 5, which support an explicit
`{"type": "disabled"}`. The "off" condition for Gemini in this study is
therefore "minimal thinking," not "verified zero thinking." Treat
cross-vendor (Claude vs. Gemini) comparisons of the on/off *effect size*
with more caution than within-vendor comparisons for exactly this reason.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any

from ..config import GEMINI_MODELS, MAX_RESPONSE_TOKENS

_GEMINI_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# The google-genai SDK does not transparently retry 429 RESOURCE_EXHAUSTED the
# way the Anthropic SDK retries its own 429s, and the free-tier Gemini quota is
# low enough that a fast full grid run trips it almost immediately (a real
# incident: 158/180 calls 429'd on the first attempt). These two knobs make a
# resumed run self-pace under a per-MINUTE limit. They cannot rescue a
# per-DAY quota — if every attempt still 429s after the full backoff, the row
# is recorded as an error and the run should be treated as blocked until the
# daily quota resets (or a billed key is used).
_THROTTLE_SECONDS = 2.0    # fixed pre-call pause to smooth requests-per-minute
_MAX_RETRIES = 6           # extra attempts on 429 before giving up on a call
_BACKOFF_BASE = 15.0       # first 429 backoff; doubles each retry
_BACKOFF_CAP = 120.0       # per-wait ceiling
_TRANSIENT_RETRIES = 2     # quick retries for non-429 errors (e.g. the
                           # intermittent 400 INVALID_ARGUMENT some Gemini
                           # models throw on a request shape that otherwise
                           # succeeds — confirmed transient, not a bad request)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _is_unsupported_thinking_level(exc: Exception) -> bool:
    """Some Gemini 3.x models (e.g. gemini-3.1-pro-preview) reject
    thinking_level="minimal" with a 400 'Thinking level MINIMAL is not
    supported for this model'. This is a deterministic capability gap, not a
    transient error — the caller responds by climbing the level ladder rather
    than retrying the same request."""
    text = str(exc).upper()
    return "THINKING LEVEL" in text and "NOT SUPPORTED" in text


# For the "off" condition we want the lowest available thinking, but not every
# model supports "minimal" — climb to "low" if minimal is rejected. "on" is
# "high" everywhere. The level that actually succeeded is recorded per row in
# GenerationResult.thinking_level_applied, so a run that mixes minimal and low
# across models stays auditable.
_OFF_LEVELS = ("minimal", "low")
_ON_LEVELS = ("high",)


@dataclass
class GenerationResult:
    model_key: str
    model_id: str
    thinking_requested: str  # "on" | "off"
    thinking_level_applied: str  # "high" | "minimal"
    response_text: str
    thinking_text: str | None
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    raw_error: str | None = None


def _resolve_api_key() -> str:
    for env_var in _GEMINI_KEY_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value
    raise RuntimeError(
        "No Gemini API key found. Set GEMINI_API_KEY (preferred) or "
        "GOOGLE_API_KEY in your environment or .env file before running the "
        "Gemini portion of this study.\n\n"
        "To get a key:\n"
        "  1. Go to https://aistudio.google.com/apikey (Google AI Studio).\n"
        "  2. Sign in and click 'Create API key'.\n"
        "  3. Copy the key into GEMINI_API_KEY in your .env file "
        "(see .env.example).\n\n"
        "This is a hard requirement, not optional — the Gemini arm of this "
        "study cannot run without it, and this harness will not silently "
        "skip Gemini models or fabricate placeholder output."
    )


_client: Any | None = None


def _get_client() -> Any:
    global _client
    if _client is None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is not installed. Install the "
                "optional Gemini dependency group with:\n"
                "  pip install -e '.[gemini]'\n"
                "from the repo root."
            ) from exc
        api_key = _resolve_api_key()
        _client = genai.Client(api_key=api_key)
    return _client


def generate(
    model_key: str,
    prompt: str,
    thinking_on: bool,
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> GenerationResult:
    """Generate one response from a Gemini-family model under one thinking
    condition. Raises RuntimeError immediately (does not return a partial
    or placeholder result) if GEMINI_API_KEY/GOOGLE_API_KEY is unset or the
    google-genai package is missing.
    """
    model_id = GEMINI_MODELS[model_key]["id"]
    # _get_client() raises a clear, actionable RuntimeError for a missing
    # API key or a missing google-genai install — call it before importing
    # `types` so those two failure modes surface their friendly messages
    # instead of a raw ModuleNotFoundError.
    client = _get_client()
    from google.genai import types

    levels = _ON_LEVELS if thinking_on else _OFF_LEVELS
    level_idx = 0
    thinking_level = levels[level_idx]

    def _build_config(level: str):
        return types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=level,
                include_thoughts=True,
            ),
        )

    config = _build_config(thinking_level)

    start = time.monotonic()
    last_exc: Exception | None = None
    response = None
    # 429s get a long exponential backoff (per-minute quota smoothing); a
    # rejected thinking level climbs the level ladder (deterministic capability
    # gap, not a retry of the same request); any other error gets a few quick
    # retries then surfaces. The error text is preserved verbatim in the record
    # either way, so nothing is masked.
    transient_used = 0
    for attempt in range(_MAX_RETRIES + 1):
        time.sleep(_THROTTLE_SECONDS)
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )
            break
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim in the record
            last_exc = exc
            if _is_unsupported_thinking_level(exc) and level_idx + 1 < len(levels):
                level_idx += 1
                thinking_level = levels[level_idx]
                config = _build_config(thinking_level)
                continue  # retry immediately at the next level (not a backoff)
            if _is_rate_limit(exc):
                if attempt == _MAX_RETRIES:
                    break
                wait = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                time.sleep(wait + random.uniform(0, 3))
            else:
                transient_used += 1
                if transient_used > _TRANSIENT_RETRIES:
                    break
                time.sleep(2.0 + random.uniform(0, 1))

    if response is None:
        return GenerationResult(
            model_key=model_key,
            model_id=model_id,
            thinking_requested="on" if thinking_on else "off",
            thinking_level_applied=thinking_level,
            response_text="",
            thinking_text=None,
            stop_reason="error",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=time.monotonic() - start,
            raw_error=str(last_exc),
        )
    latency = time.monotonic() - start

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    candidate = response.candidates[0] if response.candidates else None
    if candidate and candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            part_text = getattr(part, "text", None)
            if not part_text:
                continue
            if getattr(part, "thought", False):
                thinking_parts.append(part_text)
            else:
                text_parts.append(part_text)

    usage = response.usage_metadata
    return GenerationResult(
        model_key=model_key,
        model_id=model_id,
        thinking_requested="on" if thinking_on else "off",
        thinking_level_applied=thinking_level,
        response_text="\n".join(text_parts).strip(),
        thinking_text="\n".join(thinking_parts).strip() or None,
        stop_reason=str(candidate.finish_reason) if candidate else "",
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        latency_seconds=latency,
    )
