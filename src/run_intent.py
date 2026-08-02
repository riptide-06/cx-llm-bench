"""Intent classification eval on Banking77 (zero-shot and few-shot)."""
import argparse, difflib, json, random, re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from common import load_config, call_llm, append_raw_log

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# `datasets` >=3 dropped bare canonical names and script-based loaders, so the
# original "banking77" / "PolyAI/banking77" ids both fail. This mirror is
# parquet-native and carries the same 10003/3080 split and 77 ClassLabels.
BANKING77_REPO = "legacy-datasets/banking77"

UNPARSED = "__unparsed__"
ERRORED = "__error__"


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def build_prompt(query: str, labels: list[str], shots: list[tuple[str, str]]):
    label_block = "\n".join(f"- {l}" for l in labels)
    shot_block = ""
    if shots:
        shot_block = "Examples:\n" + "\n".join(
            f'Query: "{q}"\nIntent: {l}\n' for q, l in shots
        ) + "\n"
    return (
        "You are an intent classifier for a banking contact center.\n"
        f"Classify the customer query into exactly one of these intents:\n"
        f"{label_block}\n\n{shot_block}"
        f'Query: "{query}"\n'
        "Respond with ONLY the intent label, nothing else.\nIntent:"
    )


def parse_label(raw: str, norm_to_label: dict[str, str]) -> str:
    """Map a model's free-text answer onto a Banking77 label.

    Models wrap the answer in markdown, restate "Intent:", quote it, or add a
    trailing sentence. Matching only the bare first line scores those as
    misses and tanks accuracy, so peel the common wrappers off first and fall
    back to containment then fuzzy matching.
    """
    if not raw or not raw.strip():
        return UNPARSED

    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()

    candidates: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip("`*_ \t")
        if not line:
            continue
        line = re.sub(r"^(the\s+)?(predicted\s+|correct\s+)?intent\s*(label)?\s*[:\-]\s*",
                      "", line, flags=re.I)
        line = line.strip().strip('"\'').strip()
        line = re.sub(r"[.!,;]+$", "", line).strip()
        if line:
            candidates.append(line)
    if not candidates:
        return UNPARSED

    # 1. exact match on a normalized candidate line
    for c in candidates:
        hit = norm_to_label.get(normalize(c))
        if hit:
            return hit

    # 2. a label appearing inside a candidate (e.g. "Intent is card_arrival.")
    #    longest label first so `card_arrival` wins over a shorter substring
    norm_keys = sorted(norm_to_label, key=len, reverse=True)
    for c in candidates:
        nc = normalize(c)
        for nk in norm_keys:
            if nk in nc:
                return norm_to_label[nk]

    # 3. fuzzy match to absorb minor spelling/spacing drift
    for c in candidates:
        close = difflib.get_close_matches(normalize(c), norm_keys, n=1, cutoff=0.9)
        if close:
            return norm_to_label[close[0]]

    return UNPARSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated subset of models to run")
    # Lets an expensive model run only the cheaper condition when the budget
    # cannot cover both. A model run this way is reported at its own n for the
    # conditions it did run; it is never silently treated as complete.
    ap.add_argument("--conditions", type=str, default="zero_shot,few_shot",
                    help="comma-separated: zero_shot,few_shot")
    args = ap.parse_args()

    cfg = load_config()
    n = args.limit or cfg["intent"]["n_test"]
    k = cfg["intent"]["few_shot_k"]

    ds = load_dataset(BANKING77_REPO)
    label_names = ds["test"].features["label"].names
    norm_to_label = {normalize(l): l for l in label_names}

    rng = random.Random(SEED)
    test = list(ds["test"])
    rng.shuffle(test)
    test = test[:n]

    # few-shot pool: k examples drawn from train, fixed across all queries
    train = list(ds["train"])
    rng.shuffle(train)
    shots = [(ex["text"], label_names[ex["label"]]) for ex in train[:k]]

    models = cfg["models"]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m in wanted]

    wanted_conds = {c.strip() for c in args.conditions.split(",") if c.strip()}
    all_conds = [("zero_shot", []), ("few_shot", shots)]
    conds = [c for c in all_conds if c[0] in wanted_conds]
    if not conds:
        raise SystemExit(f"no valid conditions in --conditions {args.conditions!r}")

    results = defaultdict(lambda: defaultdict(list))
    errors = defaultdict(lambda: defaultdict(int))
    for model in models:
        for cond, cond_shots in conds:
            print(f"== {model} / {cond} ==", flush=True)
            for i, ex in enumerate(test):
                prompt = build_prompt(ex["text"], label_names, cond_shots)
                out = call_llm(model, prompt, max_tokens=30, cfg=cfg)
                if out.get("error"):
                    pred = ERRORED
                    errors[model][cond] += 1
                else:
                    pred = parse_label(out["text"], norm_to_label)
                gold = label_names[ex["label"]]
                results[model][cond].append((gold, pred))
                append_raw_log("intent", {
                    "model": model, "condition": cond, "idx": i,
                    "query": ex["text"], "gold": gold, "pred": pred,
                    "raw": out["text"], "latency_s": out["latency_s"],
                    "cost_usd": out["cost_usd"], "cached": out["cached"],
                    "error": out.get("error"),
                })
                if i % 100 == 0:
                    print(f"  {i}/{len(test)}", flush=True)
            n_err = errors[model][cond]
            if n_err:
                print(f"  [warn] {n_err}/{len(test)} calls failed for "
                      f"{model}/{cond}", flush=True)

    summary = {}
    for model, conds in results.items():
        summary[model] = {}
        for cond, pairs in conds.items():
            gold = [g for g, _ in pairs]
            pred = [p for _, p in pairs]
            # Failed API calls are excluded from accuracy (they measure the
            # provider being down, not the model being wrong) but reported so
            # the coverage loss is visible.
            scored = [(g, p) for g, p in pairs if p != ERRORED]
            g_s = [g for g, _ in scored]
            p_s = [p for _, p in scored]
            summary[model][cond] = {
                "n": len(pairs),
                "n_scored": len(scored),
                "accuracy": round(accuracy_score(g_s, p_s), 4) if scored else None,
                "macro_f1": round(f1_score(g_s, p_s, average="macro",
                                           labels=sorted(set(g_s)),
                                           zero_division=0), 4) if scored else None,
                "unparsed_rate": round(p_s.count(UNPARSED) / len(p_s), 4) if scored else None,
                "error_rate": round(pred.count(ERRORED) / len(pred), 4) if pred else None,
            }
    # MERGE rather than overwrite. Models run as separate concurrent processes
    # (see run_model.sh), and each only knows about its own model, so a plain
    # write would leave the file containing whichever process finished last.
    # Merging is also what makes it safe to run one model at a reduced set of
    # conditions without discarding another model's results.
    out_path = ROOT / "results" / "intent_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    merged = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            merged = {}
    for model, conds in summary.items():
        merged.setdefault(model, {}).update(conds)
    out_path.write_text(json.dumps(merged, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
