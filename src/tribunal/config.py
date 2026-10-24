"""Central registry: which models are in scope and where files live."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Every entry point into this package (module invocation, the scripts/
# wrappers, or an import from a notebook) goes through this module, so this
# is the one place that needs to load .env. override=True is deliberate: a
# stale ANTHROPIC_API_KEY/GEMINI_API_KEY already exported
# in the invoking shell (e.g. from an unrelated Claude Code session) must
# not silently outrank this project's own .env — that was a real incident
# during development (a run burned real API spend against the wrong key
# before this fix existed). This project's .env is the one source of truth
# for which credentials a run uses, regardless of what else is exported.
load_dotenv(override=True)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESPONSES_DIR = DATA_DIR / "responses"
FIGURES_DIR = DATA_DIR / "figures"
SUMMARY_DIR = DATA_DIR / "summary"

# --------------------------------------------------------------------------
# Claude family — three capability tiers.
#
# IDs are the exact strings from the current Anthropic model catalog
# (2026). Do not append date suffixes to the aliases below — see the
# claude-api skill's model catalog if these ever need to change.
# --------------------------------------------------------------------------

CLAUDE_MODELS = {
    "claude-haiku": {
        "id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "tier": "small",
    },
    "claude-sonnet": {
        "id": "claude-sonnet-5",
        "display_name": "Claude Sonnet 5",
        "tier": "medium",
    },
    "claude-opus": {
        "id": "claude-opus-5",
        "display_name": "Claude Opus 5",
        "tier": "large",
    },
}

# --------------------------------------------------------------------------
# Gemini family — three capability tiers.
#
# Google ships Gemini models fast and rotates preview IDs frequently. These
# were confirmed current via live web search at the time this harness was
# written (2026); if a run fails with a 404 model-not-found error, check
# https://ai.google.dev/gemini-api/docs/latest-model for the current IDs
# and update this table — do not guess a replacement ID.
# --------------------------------------------------------------------------

GEMINI_MODELS = {
    "gemini-flash-lite": {
        "id": "gemini-3.5-flash-lite",
        "display_name": "Gemini 3.5 Flash-Lite",
        "tier": "small",
    },
    "gemini-flash": {
        "id": "gemini-3.6-flash",
        "display_name": "Gemini 3.6 Flash",
        "tier": "medium",
    },
    "gemini-pro": {
        "id": "gemini-3.1-pro-preview",
        "display_name": "Gemini 3.1 Pro (preview)",
        "tier": "large",
    },
}

MAX_RESPONSE_TOKENS = 4096
