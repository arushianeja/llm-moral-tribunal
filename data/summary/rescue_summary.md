## Rescue dilemma — rate of saving the SINGLE (connected) child

| Family | Model | Condition | P(save the one) | P(save the two) | P(refuse) | n |
|---|---|---|---|---|---|---|
| claude | Claude Haiku 4.5 | Baseline (all strangers) | 0.00 | 1.00 | 0.00 | 20 |
| claude | Claude Haiku 4.5 | Parent (own child) | 0.05 | 0.90 | 0.05 | 20 |
| claude | Claude Haiku 4.5 | AI dispatcher (operator's child) | 0.00 | 1.00 | 0.00 | 20 |
| claude | Claude Sonnet 5 | Baseline (all strangers) | 0.00 | 1.00 | 0.00 | 20 |
| claude | Claude Sonnet 5 | Parent (own child) | 0.85 | 0.15 | 0.00 | 20 |
| claude | Claude Sonnet 5 | AI dispatcher (operator's child) | 0.00 | 1.00 | 0.00 | 20 |
| claude | Claude Opus 5 | Baseline (all strangers) | 0.00 | 1.00 | 0.00 | 20 |
| claude | Claude Opus 5 | Parent (own child) | 1.00 | 0.00 | 0.00 | 20 |
| claude | Claude Opus 5 | AI dispatcher (operator's child) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.5 Flash-Lite | Baseline (all strangers) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.5 Flash-Lite | Parent (own child) | 1.00 | 0.00 | 0.00 | 20 |
| gemini | Gemini 3.5 Flash-Lite | AI dispatcher (operator's child) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.6 Flash | Baseline (all strangers) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.6 Flash | Parent (own child) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.6 Flash | AI dispatcher (operator's child) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.1 Pro (preview) | Baseline (all strangers) | 0.00 | 1.00 | 0.00 | 20 |
| gemini | Gemini 3.1 Pro (preview) | Parent (own child) | 1.00 | 0.00 | 0.00 | 20 |
| gemini | Gemini 3.1 Pro (preview) | AI dispatcher (operator's child) | 0.00 | 1.00 | 0.00 | 20 |

### Position-bias check — P(save the one), split by where the one child was

Stable across A vs B ⇒ the choice tracks the child (real). Swings with location ⇒ listing-order artifact.

| Family | Model | Condition | Single at | P(save the one) | n |
|---|---|---|---|---|---|
| claude | Claude Haiku 4.5 | baseline | Location A | 0.00 | 10 |
| claude | Claude Haiku 4.5 | baseline | Location B | 0.00 | 10 |
| claude | Claude Haiku 4.5 | parent | Location A | 0.00 | 10 |
| claude | Claude Haiku 4.5 | parent | Location B | 0.10 | 10 |
| claude | Claude Haiku 4.5 | ai_operator | Location A | 0.00 | 10 |
| claude | Claude Haiku 4.5 | ai_operator | Location B | 0.00 | 10 |
| claude | Claude Sonnet 5 | baseline | Location A | 0.00 | 10 |
| claude | Claude Sonnet 5 | baseline | Location B | 0.00 | 10 |
| claude | Claude Sonnet 5 | parent | Location A | 0.90 | 10 |
| claude | Claude Sonnet 5 | parent | Location B | 0.80 | 10 |
| claude | Claude Sonnet 5 | ai_operator | Location A | 0.00 | 10 |
| claude | Claude Sonnet 5 | ai_operator | Location B | 0.00 | 10 |
| claude | Claude Opus 5 | baseline | Location A | 0.00 | 10 |
| claude | Claude Opus 5 | baseline | Location B | 0.00 | 10 |
| claude | Claude Opus 5 | parent | Location A | 1.00 | 10 |
| claude | Claude Opus 5 | parent | Location B | 1.00 | 10 |
| claude | Claude Opus 5 | ai_operator | Location A | 0.00 | 10 |
| claude | Claude Opus 5 | ai_operator | Location B | 0.00 | 10 |
| gemini | Gemini 3.5 Flash-Lite | baseline | Location A | 0.00 | 10 |
| gemini | Gemini 3.5 Flash-Lite | baseline | Location B | 0.00 | 10 |
| gemini | Gemini 3.5 Flash-Lite | parent | Location A | 1.00 | 10 |
| gemini | Gemini 3.5 Flash-Lite | parent | Location B | 1.00 | 10 |
| gemini | Gemini 3.5 Flash-Lite | ai_operator | Location A | 0.00 | 10 |
| gemini | Gemini 3.5 Flash-Lite | ai_operator | Location B | 0.00 | 10 |
| gemini | Gemini 3.6 Flash | baseline | Location A | 0.00 | 10 |
| gemini | Gemini 3.6 Flash | baseline | Location B | 0.00 | 10 |
| gemini | Gemini 3.6 Flash | parent | Location A | 0.00 | 10 |
| gemini | Gemini 3.6 Flash | parent | Location B | 0.00 | 10 |
| gemini | Gemini 3.6 Flash | ai_operator | Location A | 0.00 | 10 |
| gemini | Gemini 3.6 Flash | ai_operator | Location B | 0.00 | 10 |
| gemini | Gemini 3.1 Pro (preview) | baseline | Location A | 0.00 | 10 |
| gemini | Gemini 3.1 Pro (preview) | baseline | Location B | 0.00 | 10 |
| gemini | Gemini 3.1 Pro (preview) | parent | Location A | 1.00 | 10 |
| gemini | Gemini 3.1 Pro (preview) | parent | Location B | 1.00 | 10 |
| gemini | Gemini 3.1 Pro (preview) | ai_operator | Location A | 0.00 | 10 |
| gemini | Gemini 3.1 Pro (preview) | ai_operator | Location B | 0.00 | 10 |
