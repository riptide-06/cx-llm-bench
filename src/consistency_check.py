"""Cross-provider consistency check for llama-3.3-70b: Together vs Groq.

The 70B moved from Groq's free tier to Together's paid endpoint mid-study
because Groq's 1,000 req/day cap made it a ~52h run. Groq had already answered
a subset of prompts before the switch, so those overlapping prompts give a free
comparison of the SAME model weights served by two different stacks.

What this actually measures, and what it does not:
  - Together serves the FP8-quantized `-Turbo` build (the non-quantized build
    is not serverless on this account). Groq also serves a quantized build.
  - So a divergence is quantization + serving stack + sampling nondeterminism
    combined. It is NOT a clean "provider A vs provider B" experiment, and
    must not be written up as one.
  - Agreement here is nonetheless evidence that swapping the host mid-study did
    not silently change what was being measured, which is the thing that would
    invalidate the headline open-weights number.

Emits a markdown block for NOTES.md.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"

TOGETHER = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
GROQ = "groq/llama-3.3-70b-versatile"


def _rows(name, key_fields):
    p = RAW / f"{name}.jsonl"
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
        if r.get("error"):
            continue
        seen[tuple(r.get(k) for k in key_fields)] = r
    return seen


def compare():
    out = {}

    # --- intent: exact predicted-label agreement on shared (condition, idx) ---
    intent = _rows("intent", ("model", "condition", "idx"))
    by_model = defaultdict(dict)
    for (m, cond, idx), r in intent.items():
        by_model[m][(cond, idx)] = r
    shared = sorted(set(by_model[TOGETHER]) & set(by_model[GROQ]))
    agree = both_right = t_right = g_right = 0
    disagreements = []
    for k in shared:
        t, g = by_model[TOGETHER][k], by_model[GROQ][k]
        tp, gp, gold = t["pred"], g["pred"], t["gold"]
        if tp == gp:
            agree += 1
        else:
            disagreements.append((k, gold, tp, gp))
        t_right += tp == gold
        g_right += gp == gold
        both_right += (tp == gold) and (gp == gold)
    out["intent"] = {
        "n_shared": len(shared), "label_agreement": agree,
        "together_correct": t_right, "groq_correct": g_right,
        "both_correct": both_right, "disagreements": disagreements[:10],
    }

    # --- summarization: ROUGE-L of each provider's summary vs the reference ---
    summ = _rows("summ", ("model", "id"))
    by_m2 = defaultdict(dict)
    for (m, _id), r in summ.items():
        by_m2[m][_id] = r
    shared2 = sorted(set(by_m2[TOGETHER]) & set(by_m2[GROQ]))
    rec = {"n_shared": len(shared2)}
    if shared2:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        tl, gl, cross = [], [], []
        for i in shared2:
            t, g = by_m2[TOGETHER][i], by_m2[GROQ][i]
            tl.append(sc.score(t["reference"], t["generated"])["rougeL"].fmeasure)
            gl.append(sc.score(g["reference"], g["generated"])["rougeL"].fmeasure)
            # ROUGE-L of the two providers' outputs against EACH OTHER: how
            # textually similar the two servings are, independent of the gold
            cross.append(sc.score(g["generated"], t["generated"])["rougeL"].fmeasure)
        rec.update({
            "together_rougeL": round(sum(tl) / len(tl), 4),
            "groq_rougeL": round(sum(gl) / len(gl), 4),
            "delta": round(sum(tl) / len(tl) - sum(gl) / len(gl), 4),
            "cross_provider_rougeL": round(sum(cross) / len(cross), 4),
        })
    out["summ"] = rec
    return out


def markdown():
    c = compare()
    i, s = c["intent"], c["summ"]
    L = ["### Cross-provider consistency: llama-3.3-70b, Together vs Groq\n"]

    if not i["n_shared"] and not s["n_shared"]:
        L.append("No overlapping prompts were answered by both providers, so "
                 "no consistency check is possible.\n")
        return "\n".join(L)

    if i["n_shared"]:
        n = i["n_shared"]
        pct = 100 * i["label_agreement"] / n
        L.append(f"**Intent** — {n} prompts answered by both providers.\n")
        L.append(f"- Identical predicted label: **{i['label_agreement']}/{n} "
                 f"({pct:.1f}%)**")
        L.append(f"- Correct vs gold: Together {i['together_correct']}/{n}, "
                 f"Groq {i['groq_correct']}/{n}")
        L.append(f"- Both correct: {i['both_correct']}/{n}\n")
        if i["disagreements"]:
            L.append("Disagreements (up to 10): "
                     "`(condition, idx)` gold -> together / groq\n")
            for (cond, idx), gold, tp, gp in i["disagreements"]:
                L.append(f"- `({cond}, {idx})` `{gold}` -> `{tp}` / `{gp}`")
            L.append("")

    if s["n_shared"]:
        L.append(f"**Summarization** — {s['n_shared']} dialogues answered by both.\n")
        L.append(f"- ROUGE-L vs reference: Together **{s['together_rougeL']}**, "
                 f"Groq **{s['groq_rougeL']}** (delta {s['delta']:+.4f})")
        L.append(f"- ROUGE-L of the two providers' outputs against each other: "
                 f"**{s['cross_provider_rougeL']}**\n")

    L.append("**Interpretation.** Together serves the FP8 `-Turbo` build "
             "(the non-quantized build is not serverless on this account) and "
             "Groq also serves a quantized build, so any divergence combines "
             "quantization, serving stack and sampling nondeterminism. This is "
             "*not* a clean provider-vs-provider experiment and should not be "
             "written up as one. Its purpose is narrower: to show whether "
             "moving the 70B to a paid host mid-study changed what was being "
             "measured.\n")
    return "\n".join(L)


if __name__ == "__main__":
    print(markdown())
