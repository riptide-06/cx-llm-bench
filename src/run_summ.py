"""Customer-support dialogue summarization eval.

Primary dataset: TweetSumm (customer support dialogues + abstractive summaries),
via the DialogStudio collection. As of this run that repo is BOTH gated (needs
an HF token) and script-based (unsupported by `datasets` >=3), so it only loads
if HF_TOKEN is set in the environment. Otherwise we FALL BACK to
"knkarthick/dialogsum"; the substitution is recorded in results/NOTES.md.
"""
import argparse, json, os, random
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer

from common import load_config, call_llm, append_raw_log

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

PROMPT = (
    "You are assisting a contact center. Summarize the following customer "
    "support conversation in 2-3 sentences for an agent handoff. Capture: the "
    "customer's issue, what was tried, and the current status/next step.\n\n"
    "Conversation:\n{dialogue}\n\nSummary:"
)


def _try_tweetsumm(n: int):
    """Official TweetSumm via DialogStudio. Needs HF_TOKEN (gated repo)."""
    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")):
        raise RuntimeError("no HF_TOKEN set; TweetSumm repo is gated")
    ds = load_dataset("Salesforce/dialogstudio", "TweetSumm")["test"]
    items = []
    for i, ex in enumerate(ds):
        log = ex.get("log") or []
        turns = []
        for t in log:
            turns.append(f"Customer: {t.get('user utterance','')}")
            turns.append(f"Agent: {t.get('system response','')}")
        refs = ex.get("original dialog info", "")
        ref = ""
        if isinstance(refs, str) and refs:
            try:
                info = json.loads(refs)
                summaries = info.get("summaries", {}).get(
                    "abstractive_summaries", [])
                if summaries:
                    ref = " ".join(summaries[0])
            except Exception:
                pass
        if turns and ref:
            items.append({"id": f"tweetsumm_{i}",
                          "dialogue": "\n".join(turns), "reference": ref})
    if len(items) < n:
        raise ValueError(f"only {len(items)} usable TweetSumm items (need {n})")
    return "TweetSumm", items


def load_dialogues(n: int):
    """Returns (dataset_name, list of {id, dialogue, reference})."""
    try:
        return _try_tweetsumm(n)
    except Exception as e:
        print(f"[warn] TweetSumm load failed ({e}); falling back to dialogsum")
        ds = load_dataset("knkarthick/dialogsum")["test"]
        return "dialogsum", [
            {"id": f"dialogsum_{i}", "dialogue": ex["dialogue"],
             "reference": ex["summary"]}
            for i, ex in enumerate(ds)
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated subset of models to run")
    args = ap.parse_args()

    cfg = load_config()
    n = args.limit or cfg["summ"]["n_dialogues"]

    ds_name, items = load_dialogues(n)
    rng = random.Random(SEED)
    rng.shuffle(items)
    items = items[:n]
    print(f"Dataset: {ds_name}, {len(items)} dialogues")

    models = cfg["models"]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m in wanted]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    summary = {"dataset": ds_name, "n_dialogues": len(items), "models": {}}
    all_outputs = []
    for model in models:
        print(f"== {model} ==", flush=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        n_err = 0
        for i, item in enumerate(items):
            out = call_llm(model, PROMPT.format(dialogue=item["dialogue"]),
                           max_tokens=200, cfg=cfg)
            if out.get("error"):
                n_err += 1
            else:
                s = scorer.score(item["reference"], out["text"])
                for k in scores:
                    scores[k].append(s[k].fmeasure)
            rec = {"model": model, "id": item["id"],
                   "dialogue": item["dialogue"], "reference": item["reference"],
                   "generated": out["text"], "latency_s": out["latency_s"],
                   "cost_usd": out["cost_usd"], "cached": out["cached"],
                   "error": out.get("error")}
            append_raw_log("summ", rec)
            all_outputs.append(rec)
            if i % 25 == 0:
                print(f"  {i}/{len(items)}", flush=True)
        if n_err:
            print(f"  [warn] {n_err}/{len(items)} calls failed for {model}",
                  flush=True)
        summary["models"][model] = {
            **{k: round(sum(v) / len(v), 4) if v else None
               for k, v in scores.items()},
            "n_scored": len(scores["rougeL"]),
            "error_rate": round(n_err / len(items), 4) if items else None,
        }
    # MERGE rather than overwrite — models run as separate concurrent
    # processes and each only knows about its own model. See run_intent.py.
    out_path = ROOT / "results" / "summ_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    merged = {"dataset": ds_name, "n_dialogues": len(items), "models": {}}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("dataset") == ds_name:
                merged["models"] = prev.get("models", {})
        except json.JSONDecodeError:
            pass
    merged["models"].update(summary["models"])
    out_path.write_text(json.dumps(merged, indent=2))

    # summ_outputs.json feeds the blinded rating sheet, which must contain
    # every model in ONE batch — so merge on (model, id) here too.
    op = ROOT / "results" / "summ_outputs.json"
    prev_out = []
    if op.exists():
        try:
            prev_out = json.loads(op.read_text())
        except json.JSONDecodeError:
            prev_out = []
    by_key = {(r.get("model"), r.get("id")): r for r in prev_out}
    for r in all_outputs:
        by_key[(r["model"], r["id"])] = r
    op.write_text(json.dumps(list(by_key.values()), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
