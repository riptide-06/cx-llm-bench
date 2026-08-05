"""Paired bootstrap confidence intervals on per-dialogue ROUGE-L differences.

WHY
---
The summarization models sit within ~0.02 ROUGE-L of each other at n=100. A
ranking that tight is not obviously a ranking at all, and the paper intends to
claim the differences are "within noise". That claim needs a number.

METHOD
------
For a pair of models (A, B) scored on the SAME dialogues, compute the
per-dialogue difference d_i = ROUGE-L(A_i) - ROUGE-L(B_i). Resample the
dialogue indices with replacement B times, take the mean of d over each
resample, and read the 2.5th/97.5th percentiles as a 95% CI.

The bootstrap is PAIRED — resampling dialogue indices, not models — because
both models summarize the same dialogues, and dialogue difficulty is by far the
largest source of variance. An unpaired test would drown the model effect in
that shared variance and is the wrong instrument here.

`p_two_sided` is the fraction of resampled means that fall on the opposite side
of zero from the observed mean, doubled: an approximate two-sided bootstrap
p-value. It is not an exact test and is reported as an aid to interpretation,
not as a hypothesis test with a hard threshold.

Deterministic: fixed seed, so reruns reproduce the intervals exactly.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"

SEED = 42
N_RESAMPLES = 10_000

CORPORA = [
    ("TweetSumm", "summ_tweetsumm.jsonl"),
    ("DialogSum", "summ.jsonl"),
]


def multiref_rougeL():
    """TweetSumm per-dialogue ROUGE-L, max over all human references.

    This is the primary scoring for TweetSumm: the corpus ships ~3 human
    summaries per dialogue, and scoring against only the first measures
    agreement with one arbitrary annotator rather than summary quality.
    """
    import multiref
    per_dialogue, _means, _stats = multiref.score_rows("summ_tweetsumm.jsonl")
    return {m: {d: v["rougeL"] for d, v in dd.items()}
            for m, dd in per_dialogue.items()}


def per_dialogue_rougeL(raw_file: str):
    """-> {model: {dialogue_id: rougeL}}"""
    p = RAW / raw_file
    if not p.exists():
        return {}
    seen = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[(r["model"], r["id"])] = r  # dedupe append-only reruns

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out = defaultdict(dict)
    for (m, did), r in seen.items():
        if r.get("error"):
            continue
        gen = r.get("generated") or ""
        ref = r.get("reference") or ""
        if not gen or not ref:
            continue
        out[m][did] = sc.score(ref, gen)["rougeL"].fmeasure
    return out


def paired_bootstrap(a_scores: dict, b_scores: dict, n_resamples=N_RESAMPLES,
                     seed=SEED):
    ids = sorted(set(a_scores) & set(b_scores))
    if not ids:
        return None
    d = [a_scores[i] - b_scores[i] for i in ids]
    n = len(d)
    obs = sum(d) / n

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()

    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    # approximate two-sided bootstrap p-value
    if obs >= 0:
        p = 2.0 * sum(1 for m in means if m <= 0) / n_resamples
    else:
        p = 2.0 * sum(1 for m in means if m >= 0) / n_resamples
    return {"n_paired": n, "observed_delta": obs, "ci_low": lo, "ci_high": hi,
            "p_two_sided": min(1.0, p), "crosses_zero": lo <= 0 <= hi}


def _fmt(name_a, name_b, res, label):
    if res is None:
        return [f"- **{label}**: no shared dialogues, not computable"]
    verdict = ("**within noise** (95% CI includes 0)" if res["crosses_zero"]
               else "**distinguishable** (95% CI excludes 0)")
    return [
        f"- **{label}** — `{name_a}` minus `{name_b}`, n={res['n_paired']} paired:",
        f"  observed ΔROUGE-L = **{res['observed_delta']:+.4f}**, "
        f"95% CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], "
        f"p≈{res['p_two_sided']:.3f} → {verdict}",
    ]


def markdown(open_models: set[str] | None = None):
    lines = ["### Are the summarization differences real? Paired bootstrap\n",
             f"Paired bootstrap over per-dialogue ROUGE-L differences, "
             f"{N_RESAMPLES:,} resamples, seed {SEED}. Resampling is over "
             "dialogues (paired), because both models score the same dialogues "
             "and dialogue difficulty dominates the variance.\n"]
    variants = []
    try:
        mr = multiref_rougeL()
        if mr:
            variants.append(
                ("TweetSumm — MULTI-REFERENCE (primary; max over ~3 human "
                 "summaries)", mr))
    except Exception as e:  # twcs.csv absent -> multi-ref not computable
        variants.append((f"TweetSumm — multi-reference UNAVAILABLE ({e})", {}))
    for corpus, raw_file in CORPORA:
        label = corpus + (" — single-reference (for continuity with DialogSum)"
                          if corpus == "TweetSumm" else " — single-reference")
        variants.append((label, per_dialogue_rougeL(raw_file)))

    for corpus, scores in variants:
        if not scores:
            lines.append(f"\n**{corpus}** — not computable.\n")
            continue
        means = {m: sum(v.values()) / len(v) for m, v in scores.items() if v}
        # Only models scored on the full shared set are comparable.
        full = max((len(v) for v in scores.values()), default=0)
        means = {m: mu for m, mu in means.items() if len(scores[m]) >= full}
        ranked = sorted(means, key=lambda m: -means[m])
        lines.append(f"\n**{corpus}** (n={full} dialogues) — ranked by mean "
                     "ROUGE-L: " + ", ".join(
                         f"`{m.split('/')[-1]}` {means[m]:.4f}" for m in ranked) + "\n")
        if len(ranked) >= 2:
            a, b = ranked[0], ranked[1]
            lines += _fmt(a, b, paired_bootstrap(scores[a], scores[b]),
                          "Top-1 vs top-2")
        if open_models:
            bo = next((m for m in ranked if m in open_models), None)
            bp = next((m for m in ranked if m not in open_models), None)
            if bo and bp and bo != bp:
                lines += _fmt(bo, bp, paired_bootstrap(scores[bo], scores[bp]),
                              "Best open-weights vs best proprietary "
                              "(the compliance-gap claim)")
    return "\n".join(lines)


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    print(markdown(set(cfg.get("open_weights_models") or [])))
