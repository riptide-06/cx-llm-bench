"""Build the blinded expert rating sheet from summarization outputs.

Samples n summaries balanced across models (seed 42) into
results/expert_rating_sheet.csv. The `model` column is a blinded code; the
code -> real model mapping goes to results/expert_rating_key.csv, which must
NOT be shown to the rater.
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEED = 42


def eligible_models(outputs):
    """Which models belong in front of the expert rater.

    `summ_outputs.json` holds every model that ever produced a summary, which
    is more than should be rated:

      * `groq/llama-3.3-70b-versatile` is the cross-provider BASELINE for the
        same weights already represented by the Together entry. Showing both
        would put the same model in front of the rater twice under two blinded
        codes, contaminating the comparison.
      * Models that only ever produced a handful of summaries (Gemini managed 3
        before its daily cap) cannot fill a balanced quota, and padding the
        sheet with them would silently unbalance it.

    So: must be a reported model in config.yaml, and must have the full
    summarization set.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    reported = set(cfg.get("models") or [])
    want = cfg.get("summ", {}).get("n_dialogues", 100)
    counts = defaultdict(int)
    for r in outputs:
        counts[r["model"]] += 1
    keep, dropped = [], []
    for m in sorted(counts):
        if m in reported and counts[m] >= want:
            keep.append(m)
        else:
            why = ("not a reported model (baseline)" if m not in reported
                   else f"only {counts[m]}/{want} summaries")
            dropped.append(f"{m} — {why}")
    return keep, dropped

COLUMNS = ["sample_id", "dialogue", "model", "summary", "usefulness_1to5",
           "missing_critical_info_yn", "would_trust_in_handoff_yn", "comments"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="total summaries to sample")
    args = ap.parse_args()

    src = RES / "summ_outputs.json"
    if not src.exists():
        raise SystemExit(f"{src} not found — run src/run_summ.py first.")
    outputs = json.loads(src.read_text())

    keep, dropped = eligible_models(outputs)
    if dropped:
        print("Excluded from the rating sheet:")
        for d in dropped:
            print(f"  - {d}")
    if not keep:
        raise SystemExit("no eligible models with a full summarization set.")

    # only rateable rows: a real generated summary, no API error
    by_model = defaultdict(list)
    for r in outputs:
        if r["model"] not in keep:
            continue
        if r.get("error") or not (r.get("generated") or "").strip():
            continue
        by_model[r["model"]].append(r)
    if not by_model:
        raise SystemExit("no successful summaries available to sample.")

    models = sorted(by_model)
    rng = random.Random(SEED)

    # balanced quota, then top up round-robin if a model is short
    per = args.n // len(models)
    picked = []
    for m in models:
        pool = sorted(by_model[m], key=lambda r: r["id"])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    if len(picked) < args.n:
        chosen = {(r["model"], r["id"]) for r in picked}
        leftovers = [r for m in models for r in by_model[m]
                     if (r["model"], r["id"]) not in chosen]
        leftovers.sort(key=lambda r: (r["model"], r["id"]))
        rng.shuffle(leftovers)
        picked.extend(leftovers[:args.n - len(picked)])

    # blinding: stable pseudonyms assigned in seeded-shuffled model order
    shuffled = models[:]
    rng.shuffle(shuffled)
    code = {m: f"model_{chr(ord('A') + i)}" for i, m in enumerate(shuffled)}

    rng.shuffle(picked)  # so the rater sees no model ordering

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "expert_rating_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({
                "sample_id": f"S{i:03d}",
                "dialogue": r["dialogue"],
                "model": code[r["model"]],
                "summary": r["generated"],
                "usefulness_1to5": "",
                "missing_critical_info_yn": "",
                "would_trust_in_handoff_yn": "",
                "comments": "",
            })

    with open(RES / "expert_rating_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blinded_code", "actual_model", "n_sampled"])
        counts = defaultdict(int)
        for r in picked:
            counts[r["model"]] += 1
        for m in models:
            w.writerow([code[m], m, counts[m]])

    print(f"Wrote {RES/'expert_rating_sheet.csv'} ({len(picked)} rows, "
          f"{len(models)} models)")
    print(f"Wrote {RES/'expert_rating_key.csv'} (BLINDING KEY — withhold from rater)")


if __name__ == "__main__":
    main()
