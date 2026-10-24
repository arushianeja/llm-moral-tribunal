# Data release: rescue-dilemma study

All data were generated on 2026-08-04 (UTC, 17:19 to 18:34). License: CC BY 4.0 (see `../../DATA_LICENSE`).
Verify the headline numbers with `python3 scripts/verify_post_numbers.py` from the repo root.

## Files

| File | What it is |
|---|---|
| `rescue_clean.jsonl` | The 360 successful, parsed trials, full model text. One JSON object per line. |
| `rescue_errors_summary.csv` | Counts of FAILED attempts (165) by model, condition and error type. No raw error messages. |
| `rescue_results.json` | Compact recorded results for replay/demo use: choice plus a short rationale excerpt per trial. |
| `prompts.json` | The exact six prompts (3 conditions x 2 positions), as produced by `src/tribunal/rescue.py`. |

Design: 6 models x 3 conditions x 2 positions x 10 trials = 360 trials, all present, none missing, none duplicated.

## `rescue_clean.jsonl` fields

| Field | Type | Meaning |
|---|---|---|
| `model_key` | string | Short key: `claude-haiku`, `claude-sonnet`, `claude-opus`, `gemini-flash-lite`, `gemini-flash`, `gemini-pro`. |
| `model_id` | string | Exact API model id used (e.g. `claude-opus-5`, `gemini-3.1-pro-preview`). |
| `model_display` | string | Human-readable name. |
| `family` | string | `claude` or `gemini`. |
| `tier` | string | `small`, `medium` or `large` within the family (author's grouping). |
| `condition` | string | `baseline` (all strangers), `parent` (you are a parent; the single child is yours), `ai_operator` (you are an automated dispatcher; the single child is your operator's). |
| `single_location` | string | `A` or `B`: which location held the single child. The other held two children. Position control. |
| `trial` | int | 0-9, trial index within (model, condition, single_location). |
| `thinking_requested` | string | Always `off`. Claude: extended thinking disabled. Gemini: lowest available thinking level (minimal, falling back to low); this approximates but does not guarantee no thinking, and the level actually applied was not stored. |
| `raw_choice` | string | What the model chose, read from its `DECISION:` line: `A`, `B` or `refuse`. |
| `outcome` | string | `saved_single` (chose the location with the one child), `saved_pair` (chose the location with two), or `refuse`. Derived from `raw_choice` and `single_location`. |
| `parse_method` | string | Always `decision_line` in this release (regex on the `DECISION:` line; no fallback parse was needed). |
| `response_text` | string | The model's full reply (reasoning, then the DECISION line). |
| `generated_at` | string | ISO 8601 UTC timestamp of the call. |

Scoring is entirely by regex. There is no LLM judge anywhere in this study.

## Fields deliberately removed

`raw_error` (always empty in clean rows) and `scenario_id` (recoverable as `condition + "__single" + single_location`). Failed attempts are not included row by row: they hold only provider quota/error text, no model answers. See the CSV for counts.

## `rescue_errors_summary.csv`

Columns: `model_key`, `model_id`, `condition`, `error_type`, `failed_attempts`.
Of 525 logged calls, 165 failed (the runner is resumable and re-attempts failed calls; every cell ended with exactly 20 clean trials): 160 were `429 RESOURCE_EXHAUSTED` (free-tier rate limits, Gemini only) and 5 were `400 INVALID_ARGUMENT` (intermittent, Gemini 3.6 Flash). No Claude call failed. Failed attempts by condition: baseline 45, parent 60, ai_operator 60.

## `rescue_results.json` schema (version 1)

```
{
  "schema_version": 1,
  "description": string,
  "run_date_utc": "2026-08-04",
  "run_window_utc": [first_timestamp, last_timestamp],
  "decoding": "vendor default temperature (not set); max_tokens 4096",
  "thinking": {"claude": "disabled", "gemini": "minimal-then-low (approximate)"},
  "conditions": ["baseline", "parent", "ai_operator"],
  "models": [{"key", "id", "display", "family", "tier"}, ...],     // 6 entries
  "excerpt_max_chars": 300,
  "runs": [                                                          // 360 entries
    {"model": <model key>, "condition": <condition>, "single_location": "A"|"B",
     "trial": 0-9, "choice": "A"|"B"|"refuse",
     "outcome": "saved_single"|"saved_pair"|"refuse",
     "rationale_excerpt": string}   // reasoning text before the DECISION line,
                                    // markdown stripped, at most ~300 chars, "..." if cut
  ]
}
```

The excerpt is a convenience for display. Use `rescue_clean.jsonl` for the full text.

## Known quirks

- The prompt asks for reasoning first, then the decision, so every model produced visible reasoning even with thinking "off".
- Temperature and top_p were not set (vendor defaults); max_tokens was 4096.
- One scenario, one wording. See the main README for limitations.
