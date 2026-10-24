#!/usr/bin/env python
"""Reproduce every number quoted in the rescue-study write-up.

    python3 scripts/verify_post_numbers.py

Reads data/release/rescue_clean.jsonl (shipped in this repo). Standard library
only; needs no API key and no network. Exits 0 if all numbers match the
expected values embedded below, 1 otherwise.

Statistics are exact and dependency-free:
  * 95% Clopper-Pearson interval: bisection on the exact binomial tail.
  * Fisher exact test (two-sided): hypergeometric probabilities via
    math.comb, summing all tables as or less probable than the observed one.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "release" / "rescue_clean.jsonl"

MODELS = ["claude-haiku", "claude-sonnet", "claude-opus",
          "gemini-flash-lite", "gemini-flash", "gemini-pro"]
CONDS = ["baseline", "parent", "ai_operator"]

# Claims made in the post: (model, condition) -> (saved_single, saved_pair, refuse)
# n = 20 per cell. baseline and ai_operator are 20/20 saved_pair for all six models.
EXPECTED_PARENT = {
    "claude-haiku": (1, 18, 1),
    "claude-sonnet": (17, 3, 0),
    "claude-opus": (20, 0, 0),
    "gemini-flash-lite": (20, 0, 0),
    "gemini-flash": (0, 20, 0),
    "gemini-pro": (20, 0, 0),
}
EXPECTED_BASELINE_PAIR = 120
EXPECTED_DISPATCHER_PAIR = 120
EXPECTED_TOTAL_ROWS = 360
EXPECTED_UNPARSED = 0
EXPECTED_FALLBACK_PARSES = 0


# ----------------------------- exact statistics -----------------------------

def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    def bisect(f, lo=0.0, hi=1.0):
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # lower: smallest p with P(X >= k | p) >= alpha/2  <=>  P(X <= k-1 | p) <= 1 - alpha/2
    lower = 0.0 if k == 0 else bisect(lambda p: _binom_cdf(k - 1, n, p) > 1 - alpha / 2)
    # upper: largest p with P(X <= k | p) >= alpha/2
    upper = 1.0 if k == n else bisect(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2x2 table [[a, b], [c, d]]; two-sided exact p (probability-ordering)."""
    r1, r2, c1 = a + b, c + d, a + c
    total = r1 + r2

    def prob(x: int) -> float:
        return comb(c1, x) * comb(total - c1, r1 - x) / comb(total, r1)

    p_obs = prob(a)
    lo, hi = max(0, r1 - (total - c1)), min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


# --------------------------------- reporting --------------------------------

def fmt_ci(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {k / n:.2f}  [{lo:.3f}, {hi:.3f}]"


def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def main() -> int:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    failures: list[str] = []

    def check(label: str, got, want) -> None:
        if got != want:
            failures.append(f"{label}: got {got!r}, expected {want!r}")

    check("total clean rows", len(rows), EXPECTED_TOTAL_ROWS)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["model_key"], r["condition"])].append(r)

    print("Rescue study: reproduction from data/release/rescue_clean.jsonl")
    print(f"clean rows: {len(rows)}   (6 models x 3 conditions x 2 positions x 10 trials)\n")

    # 1. per model x condition: P(save the one), 95% Clopper-Pearson
    print("1. Saved the SINGLE child, per model x condition (k/n, 95% Clopper-Pearson)")
    print(f"{'model':<20}{'baseline':<30}{'parent':<30}{'ai_operator':<30}")
    for m in MODELS:
        line = f"{m:<20}"
        for c in CONDS:
            v = cells[(m, c)]
            check(f"n[{m},{c}]", len(v), 20)
            k = sum(1 for r in v if r["outcome"] == "saved_single")
            line += f"{fmt_ci(k, len(v)):<30}"
        print(line)

    # outcome counts (single / pair / refuse) for the parent condition
    print("\n   Parent condition outcome counts (saved_single / saved_pair / refuse)")
    for m in MODELS:
        cnt = Counter(r["outcome"] for r in cells[(m, "parent")])
        got = (cnt["saved_single"], cnt["saved_pair"], cnt["refuse"])
        check(f"parent outcomes [{m}]", got, EXPECTED_PARENT[m])
        print(f"   {m:<20}{got[0]:>3} / {got[1]:>2} / {got[2]:>2}")

    # 2. position bias
    print("\n2. Position-bias table: saved single, by where the single child was (A vs B, n=10 each)")
    print(f"{'model':<20}{'condition':<14}{'single at A':<14}{'single at B':<14}")
    for m in MODELS:
        for c in CONDS:
            v = cells[(m, c)]
            ks = {}
            for loc in "AB":
                sub = [r for r in v if r["single_location"] == loc]
                check(f"n position [{m},{c},{loc}]", len(sub), 10)
                ks[loc] = sum(1 for r in sub if r["outcome"] == "saved_single")
            print(f"{m:<20}{c:<14}{ks['A']}/10{'':<9}{ks['B']}/10")

    # 3. pooled conditions
    print("\n3. Pooled over all six models")
    for c, want_pair in (("baseline", EXPECTED_BASELINE_PAIR), ("ai_operator", EXPECTED_DISPATCHER_PAIR)):
        v = [r for m in MODELS for r in cells[(m, c)]]
        pair = sum(1 for r in v if r["outcome"] == "saved_pair")
        single = sum(1 for r in v if r["outcome"] == "saved_single")
        check(f"pooled {c} saved_pair", pair, want_pair)
        lo, hi = clopper_pearson(single, len(v))
        print(f"   {c:<12} saved the pair {pair}/{len(v)}; saved the single child {single}/{len(v)} "
              f"(95% CP upper bound {hi:.3f})")

    # 4. Fisher contrasts on the parent condition (single-child counts, n=20 each)
    print("\n4. Fisher exact (two-sided), parent condition, saved-single k/20")
    k = {m: sum(1 for r in cells[(m, "parent")] if r["outcome"] == "saved_single") for m in MODELS}
    for a, b in (("claude-haiku", "claude-sonnet"), ("claude-sonnet", "claude-opus"),
                 ("claude-haiku", "claude-opus"), ("gemini-flash", "gemini-pro")):
        p = fisher_exact_two_sided(k[a], 20 - k[a], k[b], 20 - k[b])
        print(f"   {a} ({k[a]}/20) vs {b} ({k[b]}/20): p = {fmt_p(p)}")

    # 5. refusals and parse quality
    print("\n5. Refusals and parsing")
    ref = Counter(r["model_key"] for r in rows if r["outcome"] == "refuse")
    print(f"   refusals: {sum(ref.values())} total  {dict(ref)}")
    unparsed = sum(1 for r in rows if r["outcome"] not in ("saved_single", "saved_pair", "refuse"))
    fallback = sum(1 for r in rows if r["parse_method"] != "decision_line")
    check("unparsed", unparsed, EXPECTED_UNPARSED)
    check("fallback parses", fallback, EXPECTED_FALLBACK_PARSES)
    print(f"   unparsed: {unparsed}   non-DECISION-line parses: {fallback}")

    print()
    if failures:
        print("MISMATCH with the numbers claimed in the post:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK: all numbers match the claims in the post.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
