"""Multi-reference ROUGE for TweetSumm.

WHY
---
Every TweetSumm dialogue ships **three** human abstractive summaries. Scoring
against only the first penalises a model for producing a perfectly good summary
that happens to resemble annotator 2 or 3 more than annotator 1 — it measures
agreement with one arbitrary annotator, not summary quality. With several
references available the standard convention is to score against each and take
the **maximum** per metric, which is what this module does.

This needs no new API calls: the generated summaries are already in
`results/raw/summ_tweetsumm.jsonl`. Only the extra references are new, and they
come from re-reading the TweetSumm annotations and joining on dialogue id.

Single-reference scores are retained elsewhere for continuity with the
DialogSum comparison (DialogSum has one reference per dialogue, so it cannot be
scored this way and the two corpora must be compared single-ref to single-ref).
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"

METRICS = ("rouge1", "rouge2", "rougeL")


@lru_cache(maxsize=1)
def reference_map() -> dict[str, list[str]]:
    """dialogue id -> all abstractive references (requires twcs.csv locally)."""
    import tweetsumm_loader
    items, _ = tweetsumm_loader.load_split("test")
    return {it["id"]: it["references"] for it in items}


def _dedupe(raw_file: str):
    p = RAW / raw_file
    if not p.exists():
        return []
    seen = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[(r["model"], r["id"])] = r
    return list(seen.values())


def score_rows(raw_file: str = "summ_tweetsumm.jsonl"):
    """-> (per_dialogue, per_model_means, stats)

    per_dialogue: {model: {dialogue_id: {metric: max_over_refs}}}
    """
    refs_by_id = reference_map()
    sc = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)

    per_dialogue = defaultdict(dict)
    n_refs_used = []
    missing_refs = 0

    for r in _dedupe(raw_file):
        if r.get("error"):
            continue
        gen = (r.get("generated") or "").strip()
        if not gen:
            continue
        refs = refs_by_id.get(r["id"])
        if not refs:
            # Fall back to the single stored reference rather than dropping the
            # row; counted so the fallback can never be silent.
            single = (r.get("reference") or "").strip()
            if not single:
                continue
            refs = [single]
            missing_refs += 1
        n_refs_used.append(len(refs))

        best = {m: 0.0 for m in METRICS}
        for ref in refs:
            s = sc.score(ref, gen)
            for m in METRICS:
                best[m] = max(best[m], s[m].fmeasure)
        per_dialogue[r["model"]][r["id"]] = best

    means = {}
    for model, d in per_dialogue.items():
        if not d:
            continue
        means[model] = {
            **{m: round(sum(v[m] for v in d.values()) / len(d), 4)
               for m in METRICS},
            "n_scored": len(d),
        }

    stats = {
        "references_per_dialogue": (
            round(sum(n_refs_used) / len(n_refs_used), 2) if n_refs_used else 0),
        "rows_without_multiref_fallback_to_single": missing_refs,
        "models": len(means),
    }
    return dict(per_dialogue), means, stats


def summary_json(raw_file: str = "summ_tweetsumm.jsonl"):
    """Same shape as the single-ref summaries, so report.py can consume it."""
    per_dialogue, means, stats = score_rows(raw_file)
    models = {}
    for m, v in means.items():
        models[m] = {**v, "error_rate": 0.0}
    n = max((v["n_scored"] for v in means.values()), default=0)
    return {
        "dataset": "TweetSumm",
        "scoring": "multi-reference (max over 3 human abstractive summaries)",
        "n_dialogues": n,
        "references_per_dialogue": stats["references_per_dialogue"],
        "models": models,
    }, per_dialogue, stats


if __name__ == "__main__":
    s, per_dialogue, stats = summary_json()
    print(json.dumps({k: v for k, v in s.items() if k != "models"}, indent=2))
    print(json.dumps(stats, indent=2))
    print()
    for m, v in sorted(s["models"].items(), key=lambda x: -x[1]["rougeL"]):
        print(f"  {m[:46]:46s} n={v['n_scored']:3d} "
              f"R1={v['rouge1']:.4f} R2={v['rouge2']:.4f} RL={v['rougeL']:.4f}")
