"""Rebuild the summary JSONs from results/raw/*.jsonl — the authoritative record.

WHY THIS EXISTS
---------------
`intent_summary.json`, `summ_summary.json` and `summ_outputs.json` are written
by the runners, but models run as separate concurrent processes and each
process only knows about its own model. Before merge-on-write was added, later
processes clobbered earlier ones — `summ_outputs.json` ended up holding 2 of 6
models, which would have produced a silently incomplete expert rating sheet.

Re-running the runners to regenerate them is NOT a safe fix: for a model that
stopped early (Gemini, capped at 500 requests/day) the uncached remainder would
fire as live calls and fail, and for a model still running it would double that
model's request rate against its own quota.

The raw logs already contain every field needed to recompute every metric, and
rebuilding from them is deterministic, idempotent, makes no API calls, and
handles partially-completed models naturally. Metric definitions here mirror
run_intent.py / run_summ.py exactly.

Run any time; safe to run while other models are still going.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RAW = RES / "raw"

UNPARSED = "__unparsed__"
ERRORED = "__error__"


def _dedupe(name, key_fields):
    """Raw logs are append-only and reruns re-append; keep one row per call."""
    p = RAW / f"{name}.jsonl"
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
        seen[tuple(r.get(k) for k in key_fields)] = r
    return list(seen.values())


def rebuild_intent():
    rows = _dedupe("intent", ("model", "condition", "idx"))
    pairs = defaultdict(list)
    for r in rows:
        pred = ERRORED if r.get("error") else r.get("pred")
        pairs[(r["model"], r["condition"])].append((r.get("gold"), pred))

    out = defaultdict(dict)
    for (model, cond), ps in pairs.items():
        gold = [g for g, _ in ps]
        pred = [p for _, p in ps]
        # Failed API calls measure a provider being down, not a wrong answer,
        # so they are excluded from accuracy but surfaced via error_rate.
        scored = [(g, p) for g, p in ps if p != ERRORED]
        g_s = [g for g, _ in scored]
        p_s = [p for _, p in scored]
        out[model][cond] = {
            "n": len(ps),
            "n_scored": len(scored),
            "accuracy": round(accuracy_score(g_s, p_s), 4) if scored else None,
            "macro_f1": round(f1_score(g_s, p_s, average="macro",
                                       labels=sorted(set(g_s)),
                                       zero_division=0), 4) if scored else None,
            "unparsed_rate": round(p_s.count(UNPARSED) / len(p_s), 4) if scored else None,
            "error_rate": round(pred.count(ERRORED) / len(pred), 4) if pred else None,
        }
    return dict(out)


def rebuild_summ(raw_name="summ", dataset_label="dialogsum"):
    rows = _dedupe(raw_name, ("model", "id"))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    per = defaultdict(lambda: {"rouge1": [], "rouge2": [], "rougeL": [],
                               "n": 0, "err": 0})
    outputs = []
    for r in rows:
        m = r["model"]
        s = per[m]
        s["n"] += 1
        if r.get("error"):
            s["err"] += 1
        else:
            sc = scorer.score(r.get("reference") or "", r.get("generated") or "")
            for k in ("rouge1", "rouge2", "rougeL"):
                s[k].append(sc[k].fmeasure)
        outputs.append(r)

    models = {}
    for m, s in per.items():
        models[m] = {
            **{k: round(sum(s[k]) / len(s[k]), 4) if s[k] else None
               for k in ("rouge1", "rouge2", "rougeL")},
            "n_scored": len(s["rougeL"]),
            "error_rate": round(s["err"] / s["n"], 4) if s["n"] else None,
        }
    n_dialogues = max((v["n_scored"] for v in models.values()), default=0)
    return {"dataset": dataset_label, "n_dialogues": n_dialogues,
            "models": models}, outputs


def main():
    intent = rebuild_intent()
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "intent_summary.json").write_text(json.dumps(intent, indent=2))

    print("intent_summary.json:")
    for m, c in sorted(intent.items()):
        for cond, v in sorted(c.items()):
            print(f"  {m[:44]:44s} {cond:10s} n={v['n']:5d} acc={v['accuracy']}")

    # Both summarization corpora: TweetSumm is the primary result (in-domain
    # customer support), DialogSum the secondary cross-domain robustness check.
    for raw_name, label, summ_file, out_file in (
            ("summ_tweetsumm", "TweetSumm",
             "summ_tweetsumm_summary.json", "tweetsumm_outputs.json"),
            ("summ", "dialogsum", "summ_summary.json", "summ_outputs.json")):
        if not (RAW / f"{raw_name}.jsonl").exists():
            print(f"{summ_file}: (no raw log, skipped)")
            continue
        summ, outputs = rebuild_summ(raw_name, label)
        (RES / summ_file).write_text(json.dumps(summ, indent=2))
        (RES / out_file).write_text(json.dumps(outputs, indent=2))
        print(f"{summ_file}  [{label}]:")
        for m, v in sorted(summ["models"].items()):
            print(f"  {m[:44]:44s} n={v['n_scored']:4d} rougeL={v['rougeL']}")
        print(f"  -> {out_file}: {len(outputs)} rows")


if __name__ == "__main__":
    main()
