"""Cost per 1,000 calls at published hosted-API list prices.

WHY THIS EXISTS
---------------
Four of the five models in this study ran on free promotional tiers, so what we
were actually billed is $0.00 for them and ~$2 for claude-haiku. Plotting that
would say "open-weights models are free", which is an artifact of promotional
pricing, not an economic finding. This module instead prices every call at each
provider's PUBLISHED list price, so the cost axis compares like with like.

The axis this produces is *hosted API list price*. It is NOT self-hosting TCO
(GPU capital/rental, engineering time, idle capacity) — that belongs in the
paper's limitations, not on this chart.

METHOD
------
The cache stores responses but not token counts, and the providers' `usage`
fields were not persisted, so token counts are ESTIMATED:

  * Input  — prompts are reconstructed EXACTLY as the runners built them
             (`run_intent.build_prompt`, `run_summ.PROMPT`) from the query /
             dialogue text preserved in `results/raw/*.jsonl`, then tokenized.
  * Output — the generated text stored in the same raw logs, tokenized.

Tokenization uses `litellm.token_counter(model=...)`, which selects a
model-appropriate tokenizer where it knows one and otherwise falls back to a
cl100k-style default. For the non-OpenAI models this is an approximation:
tokenizer differences of a few percent are expected, and the estimate excludes
provider-side additions such as chat-template scaffolding and BOS/EOS tokens.
It is deterministic and applied identically to every model, so it is sound for
*comparing* models, which is what the chart is for. Absolute values should be
quoted as estimates.

Prices are recorded in PRICES below with source URL and retrieval date.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import litellm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"

PRICE_RETRIEVED = "2026-07-31"

# USD per 1,000,000 tokens. `snapshot` is the concrete model the id served, and
# is what the price was looked up for.
PRICES = {
    "claude-haiku-4-5-20251001": {
        "snapshot": "claude-haiku-4-5-20251001",
        "input_per_m": 1.00, "output_per_m": 5.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing (Claude Haiku 4.5)",
    },
    "claude-sonnet-5": {
        # NOTE FOR THE PAPER: $2/$10 is INTRODUCTORY pricing, in effect only
        # through 2026-08-31. From 2026-09-01 Sonnet 5 lists at $3/$15, which
        # would raise its cost-per-1k here by 50%. Anyone reproducing this
        # after that date must re-derive the cost axis; the retrieval date
        # below is load-bearing, not decorative.
        "snapshot": "claude-sonnet-5",
        "input_per_m": 2.00, "output_per_m": 10.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing "
                  "(Claude Sonnet 5, introductory rate through 2026-08-31)",
    },
    "gemini/gemini-flash-lite-latest": {
        # The alias reports itself as gemini-3.5-flash-lite in the response
        # `modelVersion` field. Priced as 3.5 Flash-Lite accordingly - NOT as
        # 2.5 Flash-Lite ($0.10/$0.40), which is what litellm's bundled pricing
        # DB assumes and which would understate this model ~3x/6x.
        "snapshot": "gemini-3.5-flash-lite",
        "input_per_m": 0.30, "output_per_m": 2.50,
        "source": "https://ai.google.dev/gemini-api/docs/pricing (Gemini 3.5 Flash-Lite)",
    },
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        # Published list price on together.ai/pricing. litellm 1.94.1's bundled
        # DB says $0.88/$0.88, so `billed_cost_per_1k` for this model is a
        # litellm estimate at the older rate, not an invoice. The list-price
        # axis uses the published $1.04.
        "snapshot": "meta-llama/Llama-3.3-70B-Instruct-Turbo (FP8)",
        "input_per_m": 1.04, "output_per_m": 1.04,
        "source": "https://www.together.ai/pricing (Llama 3.3 70B)",
    },
    "groq/llama-3.3-70b-versatile": {
        # Consistency-check baseline only; not a reported model.
        "snapshot": "llama-3.3-70b-versatile",
        "input_per_m": 0.59, "output_per_m": 0.79,
        "source": "https://groq.com/pricing (Llama 3.3 70B Versatile 128k)",
    },
    "groq/llama-3.1-8b-instant": {
        "snapshot": "llama-3.1-8b-instant",
        "input_per_m": 0.05, "output_per_m": 0.08,
        "source": "https://groq.com/pricing (Llama 3.1 8B Instant 128k)",
    },
    "mistral/mistral-small-latest": {
        "snapshot": "mistral-small-latest (Mistral Small 4)",
        "input_per_m": 0.15, "output_per_m": 0.60,
        "source": "https://mistral.ai/pricing/api (Mistral Small 4)",
    },
}


def _count(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        # last-resort fallback so one unknown id cannot break the whole report
        return max(1, len(text) // 4)


def _dedupe(path: Path, key_fields):
    """Raw logs are append-only and reruns re-append; keep one row per call."""
    if not path.exists():
        return []
    seen = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[tuple(r.get(k) for k in key_fields)] = r
    return list(seen.values())


def list_cost_per_1k():
    """-> {(model, condition): {...}} priced at published list prices.

    condition is the intent condition ("zero_shot"/"few_shot") or "all" for
    summarization, matching how report.py keys its cost/latency merge.
    """
    # Imported lazily: run_intent pulls in datasets/sklearn, and report.py
    # should still work if those are unavailable.
    from run_intent import build_prompt
    from run_summ import PROMPT as SUMM_PROMPT
    from datasets import load_dataset
    import random

    # Rebuild the exact label list and few-shot exemplars the runner used, so
    # reconstructed prompts are token-identical to the ones actually sent.
    ds = load_dataset("legacy-datasets/banking77")
    label_names = ds["test"].features["label"].names
    rng = random.Random(42)
    test = list(ds["test"])
    rng.shuffle(test)
    train = list(ds["train"])
    rng.shuffle(train)
    shots = [(ex["text"], label_names[ex["label"]]) for ex in train[:5]]

    agg = defaultdict(lambda: {"in": 0, "out": 0, "n": 0})

    for r in _dedupe(RAW / "intent.jsonl", ("model", "condition", "idx")):
        if r.get("error"):
            continue
        m, cond = r["model"], r["condition"]
        prompt = build_prompt(r["query"], label_names,
                              shots if cond == "few_shot" else [])
        a = agg[(m, cond)]
        a["in"] += _count(m, prompt)
        a["out"] += _count(m, r.get("raw") or "")
        a["n"] += 1

    # The two summarization corpora are priced SEPARATELY — TweetSumm dialogues
    # average ~1,280 chars against DialogSum's ~700, so blending them under one
    # key would misstate cost-per-1k for both. Keys: "all" keeps the DialogSum
    # key stable for existing consumers; TweetSumm gets its own.
    for fname, cond_key in (("summ.jsonl", "all"),
                            ("summ_tweetsumm.jsonl", "tweetsumm")):
        for r in _dedupe(RAW / fname, ("model", "id")):
            if r.get("error"):
                continue
            m = r["model"]
            a = agg[(m, cond_key)]
            a["in"] += _count(m, SUMM_PROMPT.format(dialogue=r["dialogue"]))
            a["out"] += _count(m, r.get("generated") or "")
            a["n"] += 1

    out = {}
    for (m, cond), a in agg.items():
        if not a["n"]:
            continue
        p = PRICES.get(m)
        mean_in, mean_out = a["in"] / a["n"], a["out"] / a["n"]
        rec = {
            "mean_input_tokens": round(mean_in, 1),
            "mean_output_tokens": round(mean_out, 1),
            "n_priced_calls": a["n"],
        }
        if p:
            rec["list_cost_per_1k"] = round(
                (mean_in * p["input_per_m"] + mean_out * p["output_per_m"])
                / 1e6 * 1000, 4)
        else:
            rec["list_cost_per_1k"] = None  # no published price on file
        out[(m, cond)] = rec
    return out


def price_table_markdown():
    rows = ["| Model | Priced as | Input $/M | Output $/M | Source |",
            "|---|---|---:|---:|---|"]
    for m, p in PRICES.items():
        rows.append(f"| `{m}` | `{p['snapshot']}` | {p['input_per_m']:.2f} | "
                    f"{p['output_per_m']:.2f} | {p['source']} |")
    rows.append("")
    rows.append(f"All prices retrieved **{PRICE_RETRIEVED}**.")
    return "\n".join(rows)


if __name__ == "__main__":
    print(price_table_markdown())
    print()
    for k, v in sorted(list_cost_per_1k().items()):
        print(k, v)
