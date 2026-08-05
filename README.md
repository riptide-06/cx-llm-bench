# CX-LLM-Bench

**Benchmarking open-weights vs. proprietary LLMs for privacy-constrained
contact center automation**

Reproducibility artifact for the paper *"Closing the Compliance Gap:
Benchmarking Open-Weights vs. Proprietary LLMs for Privacy-Constrained Contact
Center Automation"* (in preparation, target: *Computers*, MDPI).

---

## Summary

Healthcare and other privacy-constrained enterprises often cannot send customer
conversations to a proprietary API, restricting them to self-hostable
open-weights models. This benchmark measures what that restriction actually
costs in task performance. Six models — two proprietary (Claude Haiku 4.5,
Claude Sonnet 5) and three open-weights (Llama-3.3-70B, Mistral Small,
Llama-3.1-8B), plus a capped Gemini subsample — were evaluated on **intent
classification** (Banking77, n=1,000, zero-shot *and* few-shot k=5) and
**conversation summarization** (DialogSum, n=100), across **10,156 API calls**.

The best open-weights model trails the best proprietary model by **7.56%
relative accuracy zero-shot** and **6.66% few-shot** (10.48% / 8.55% on
macro-F1, which weights Banking77's 77 classes equally and is arguably the more
decision-relevant metric for routing). At published API list prices the
open-weights side is **~2.7x cheaper** ($0.52 vs $1.39 per 1,000 zero-shot
queries).

**On summarization the ordering inverts.** On TweetSumm — genuine
customer-support dialogue, scored **multi-reference** (max over its ~3 human
summaries per dialogue) — Llama-3.3-70B *beats* the best proprietary model by
**8.26% relative ROUGE-L** (0.2738 vs 0.2529). A paired bootstrap (10,000
resamples over the 100 shared dialogues) puts that at **+0.0209, 95% CI
[+0.0099, +0.0316], p≈0.000** — the interval excludes zero, so it is not noise.
The result holds under single-reference scoring too (+0.0138, p≈0.009) and gets
*stronger* multi-reference, so it is not an artifact of the metric choice.

The same comparison on DialogSum (general-domain, secondary, single-reference)
gives **+0.0074, 95% CI [−0.0052, +0.0201], p≈0.245** — *within* noise. So
summarization rankings here are **corpus-specific** and do not transfer across
domains; Llama-3.3-70B is 1st of 6 on TweetSumm and last on DialogSum. See
`results/NOTES.md` §11.

**These gaps are upper bounds.** The 70B was served FP8-quantized, and
quantization can only depress the open-weights score; since
`gap = proprietary − open`, a depressed `open` inflates the gap. FP8 can
overstate the gap, never understate it.

Full results: [`results/tables.md`](results/tables.md) ·
Methods-level detail: [`results/NOTES.md`](results/NOTES.md) ·
Run provenance and cost: [`results/RUN_REPORT.md`](results/RUN_REPORT.md)

![accuracy vs cost](results/charts/accuracy_vs_cost.png)

---

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in your API keys
```

### Getting the summarization data (required for the primary result)

The primary summarization corpus is **TweetSumm**, which distributes only tweet
**IDs** — the tweet **text** must come from Kaggle. That is a licensing
requirement (CDLA-Sharing-1.0), so this repository ships the loader and nothing
else; no dialogue text is redistributed here.

```bash
git clone https://github.com/guyfe/Tweetsumm.git vendor/tweetsumm

# Download "Customer Support on Twitter" from Kaggle, then:
#   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
mkdir -p data && mv /path/to/twcs.csv data/twcs.csv

python src/tweetsumm_loader.py    # verify reconstruction
```

That check should report `tweet_ids_found_in_twcs` equal to
`tweet_ids_referenced` and `dialogues_skipped_missing_tweets: 0`. Both `data/`
and `vendor/` are gitignored. The secondary DialogSum corpus downloads
automatically from the Hub and needs no setup.

Keys are read from `.env` by `src/common.py`. **A key exported in your
interactive shell will not reach a spawned run** — put it in `.env`.

Run in this order:

```bash
python src/run_intent.py                       # Banking77: zero-shot + few-shot
python src/run_summ.py --dataset tweetsumm     # PRIMARY summarization
python src/run_summ.py --dataset dialogsum     # secondary robustness check
python src/rebuild_summaries.py                # rebuild summaries from raw logs
python src/report.py                           # -> results/tables.md + charts
python src/make_rating_sheet.py                # blinded expert rating sheet
```

Useful flags:

- `--limit N` — smoke-test with a small sample
- `--models a,b` — run a subset; **each model should run in its own process**,
  because rate limiting is per-model *within* a process (see below)
- `--conditions zero_shot` (intent only) — run one condition when the budget
  cannot cover both

Every call is cached in `cache/` keyed on `(model, prompt, max_tokens)`, so
reruns are free and interrupted runs resume exactly where they stopped. Failed
calls are never cached, so a rerun retries them.

Optional: `src/consistency_check.py` compares the same open-weights model served
by two different providers.

---

## The rate-limit lesson: budget the whole workload against the *daily* cap

The single most costly mistake in this project, and it is not the one it first
appeared to be.

`llama-3.1-8b-instant` completed its 1,000 zero-shot calls and then crawled at
~120 s/call on few-shot, eventually being abandoned at ~32 h projected. This was
initially diagnosed as a tokens-per-*minute* problem and "fixed" by increasing
the per-request delay. The provider's own error message says otherwise:

```
Rate limit reached ... on tokens per day (TPD):
Limit 500000, Used 499554, Requested 684.
```

Groq's free tier meters that model at **500,000 tokens per day**. The zero-shot
arm — 1,000 calls at ~500 prompt tokens — had spent essentially the entire daily
budget. The slow crawl afterwards was the bucket *refilling* at ~5.8 tokens/sec,
not backoff thrash. **No per-request delay recovers a budget that is already
spent.**

What to actually do:

1. **Compute `n_calls x prompt_tokens` against the daily token cap before you
   start.** Here that was 2,100 x ~570 ≈ 1.2M tokens against 500k/day: the
   workload never fit in one day, and no pacing could have changed that.
2. **Order conditions by cost when they share a budget.** Running the cheap arm
   first consumed the whole day.
3. **Per-minute pacing is a separate constraint.** Both matter; don't conflate
   them. Read the `x-ratelimit-*` response headers rather than trusting the
   client library — daily caps in particular are often absent from it.

Two related traps documented in `results/NOTES.md`:

- **Request-per-day caps bite too.** Groq allows 1,000 requests/day for
  `llama-3.3-70b-versatile` (~52 h for this workload, which forced a move to a
  paid host); Gemini's free tier caps `gemini-3.5-flash-lite` at 500/day.
- **Thinking models cannot answer under a small `max_tokens`.** Several
  otherwise-capable models spend the entire completion budget reasoning and
  return a truncated fragment, which scores as ~0 accuracy and looks like a
  parsing bug. They were excluded rather than reported as broken scores.

---

## Cost figures are list price, not what we paid

Four of the six models ran on **free promotional tiers**, so billed cost is ~$0
for them. Plotting that would make the cost axis an artifact of promotions, so
`report.py` prices every call at its provider's **published list price**
(`src/list_price.py`, with sources and retrieval dates).

Two caveats worth repeating:

- This is **hosted API list price**, *not* self-hosting TCO. GPU capital or
  rental, engineering time, and idle capacity are the real costs a
  privacy-constrained enterprise trades against, and none are on this axis.
- Client-library price tables can be stale. Three of six models were mispriced
  in the bundled table of the version used (one by 3x/6.25x); all prices here
  were verified against vendor pages. **Re-derive before reusing** — one model's
  rate was introductory pricing with a known expiry.

---

## Repository layout

```
src/                     benchmark, reporting and analysis code
config.yaml              model lineup, rate limits, sample sizes
results/tables.md        per-model metrics + the compliance-gap table
results/NOTES.md         Methods-level detail: datasets, model IDs, rate
                         limits, pricing provenance, threats to validity
results/RUN_REPORT.md    cost, call counts, every substitution and failure
results/charts/          accuracy-vs-cost, ROUGE-L-vs-cost, zero vs few-shot
results/raw/             call-level record (10,156 calls) — the data artifact
CLAUDE.md                original project brief
```

**Not in this repository:**

- `cache/` — ~10k cached API responses, too heavy to commit and fully
  reproducible from the code. **Available on request.**
- `results/expert_rating_sheet.csv` and its blinding key — human expert
  evaluation is in progress; anonymized results will be added after review.
- `results/summ_outputs.json` and `results/raw/summ.jsonl` — **held back for the
  same reason**: both map each generated summary to the model that produced it,
  which would de-blind the in-progress expert evaluation. Aggregate
  summarization metrics are public in `results/tables.md`; the per-call records
  are **available on request** and will be released with the rating results.
  `results/raw/intent.jsonl` (the 4,000-call intent record) is published in
  full — it carries no blinding concern.
- `results/run_logs/` — raw provider stdout, build noise rather than results.
- `paper/` — manuscript draft, held back until authorship is finalized.

---

## Reproducibility notes

- `temperature=0` and `seed=42` throughout (with `drop_params` for providers
  that reject `seed`). Determinism is *not* guaranteed across providers: the
  same open-weights model served by two hosts produced summaries with a
  cross-provider ROUGE-L of 0.639 — substantially different wording — while
  scoring within 0.002 of each other against the reference.
- Datasets: Banking77 via `legacy-datasets/banking77` (test split, seed-42
  shuffle, first 1,000). Summarization primary: **TweetSumm** test split
  (110 dialogues, all 1,152 referenced tweet IDs resolved, seed-42 shuffle,
  first 100), reconstructed locally from tweet IDs + Kaggle `twcs.csv` using the
  upstream `TweetSumProcessor`; the **first** of each dialogue's three human
  abstractive annotations is the reference. Summarization secondary:
  `knkarthick/dialogsum` (test split, seed-42 shuffle, first 100), retained as a
  cross-domain robustness check — it is general-domain daily conversation, not
  customer support.
- **Licensing:** TweetSumm's dataset is CDLA-Sharing-1.0 and its tweet text
  comes from Kaggle. This repository therefore contains **no** dialogue text,
  no `twcs.csv`, and no summarization records derived from it — only the loader
  and the IDs. Reproducers fetch the text from Kaggle themselves.
- Moving aliases (`mistral-small-latest`, `gemini-flash-lite-latest`) may
  hot-swap. Concrete snapshots served during this run are recorded in
  `results/NOTES.md` §10 — cite those, not the aliases.
- Sample sizes were reduced from 1,500/120 to 1,000/100 to fit free-tier daily
  request caps. The reduction is cache-safe: the test split is shuffled *before*
  slicing, so the smaller set is a valid random subsample.

---

## Citation

```bibtex
@article{nagaraju2026compliancegap,
  title   = {Closing the Compliance Gap: Benchmarking Open-Weights vs.
             Proprietary LLMs for Privacy-Constrained Contact Center Automation},
  author  = {Nagaraju, Tarun and others},
  journal = {Computers},
  year    = {2026},
  note    = {In preparation. Artifact: https://github.com/riptide-06/cx-llm-bench}
}
```

## License

MIT — see [LICENSE](LICENSE).
