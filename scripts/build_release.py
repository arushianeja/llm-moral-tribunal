#!/usr/bin/env python
"""Build the public data release from the raw rescue run (maintainer tool).

Reads  data/responses/rescue_responses.jsonl  (raw, gitignored: contains
failed-attempt rows with provider error text) and writes, under data/release/:

    rescue_clean.jsonl            the clean rows only, sensitive/irrelevant
                                  fields removed
    rescue_errors_summary.csv     counts of failed attempts, no raw messages
    rescue_results.json           compact recorded-results file (see
                                  data/release/README.md for the schema)

Standard library only. No network, no API keys. Anyone who just wants to
verify the numbers does NOT need to run this: the outputs are committed.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "responses" / "rescue_responses.jsonl"
OUT = ROOT / "data" / "release"

DISPLAY = {
    "claude-haiku": ("Claude Haiku 4.5", "claude", "small"),
    "claude-sonnet": ("Claude Sonnet 5", "claude", "medium"),
    "claude-opus": ("Claude Opus 5", "claude", "large"),
    "gemini-flash-lite": ("Gemini 3.5 Flash-Lite", "gemini", "small"),
    "gemini-flash": ("Gemini 3.6 Flash", "gemini", "medium"),
    "gemini-pro": ("Gemini 3.1 Pro (preview)", "gemini", "large"),
}
MODEL_ORDER = list(DISPLAY)
COND_ORDER = ["baseline", "parent", "ai_operator"]
EXCERPT_CHARS = 300
_DECISION = re.compile(r"\n?[^\n]*DECISION:.*$", re.IGNORECASE | re.DOTALL)


def excerpt(text: str) -> str:
    """Reasoning text before the DECISION line, markdown headers removed,
    whitespace collapsed, cut at a word boundary to <= EXCERPT_CHARS."""
    body = _DECISION.sub("", text)
    body = re.sub(r"(?m)^\s*#+\s*(Reasoning|Decision)\s*$", "", body)
    body = re.sub(r"[*_`#]+", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) <= EXCERPT_CHARS:
        return body
    cut = body[:EXCERPT_CHARS].rsplit(" ", 1)[0]
    return cut + "..."


def error_type(msg: str) -> str:
    m = re.match(r"\s*(\d{3})\s+([A-Z_]+)", msg)
    return f"{m.group(1)} {m.group(2)}" if m else "other"


def main() -> None:
    rows = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
    clean = [r for r in rows if not r.get("raw_error")]
    failed = [r for r in rows if r.get("raw_error")]
    OUT.mkdir(parents=True, exist_ok=True)

    clean.sort(key=lambda r: (MODEL_ORDER.index(r["model_key"]),
                              COND_ORDER.index(r["condition"]),
                              r["single_location"], int(r["trial"])))
    with open(OUT / "rescue_clean.jsonl", "w") as f:
        for r in clean:
            name, fam, tier = DISPLAY[r["model_key"]]
            rec = {
                "model_key": r["model_key"],
                "model_id": r["model_id"],
                "model_display": name,
                "family": fam,
                "tier": tier,
                "condition": r["condition"],
                "single_location": r["single_location"],
                "trial": int(r["trial"]),
                "thinking_requested": r["thinking_requested"],
                "raw_choice": r["raw_choice"],
                "outcome": r["outcome"],
                "parse_method": r["parse_method"],
                "response_text": r["response_text"],
                "generated_at": r["generated_at"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cnt = Counter((r["model_key"], r["condition"], error_type(r["raw_error"])) for r in failed)
    with open(OUT / "rescue_errors_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_key", "model_id", "condition", "error_type", "failed_attempts"])
        ids = {r["model_key"]: r["model_id"] for r in rows}
        for (mk, cond, et), n in sorted(cnt.items(), key=lambda kv: (
                MODEL_ORDER.index(kv[0][0]), COND_ORDER.index(kv[0][1]), kv[0][2])):
            w.writerow([mk, ids[mk], cond, et, n])

    dates = sorted(r["generated_at"] for r in clean)
    results = {
        "schema_version": 1,
        "description": "Recorded outputs of the rescue-dilemma study, for replay. "
                       "Nothing here calls a model.",
        "run_date_utc": dates[0][:10],
        "run_window_utc": [dates[0], dates[-1]],
        "decoding": "vendor default temperature (not set); max_tokens 4096",
        "thinking": {"claude": "disabled", "gemini": "minimal-then-low (approximate)"},
        "conditions": COND_ORDER,
        "models": [{"key": k, "id": next(r["model_id"] for r in clean if r["model_key"] == k),
                    "display": DISPLAY[k][0], "family": DISPLAY[k][1], "tier": DISPLAY[k][2]}
                   for k in MODEL_ORDER],
        "excerpt_max_chars": EXCERPT_CHARS,
        "runs": [],
    }
    for r in clean:
        results["runs"].append({
            "model": r["model_key"],
            "condition": r["condition"],
            "single_location": r["single_location"],
            "trial": int(r["trial"]),
            "choice": r["raw_choice"],
            "outcome": r["outcome"],
            "rationale_excerpt": excerpt(r["response_text"]),
        })
    (OUT / "rescue_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")
    print(f"clean rows: {len(clean)}  failed attempts: {len(failed)}  -> {OUT}")


if __name__ == "__main__":
    main()
