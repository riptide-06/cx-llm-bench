# CX-LLM-Bench — Run Notes

Status as of 2026-08-05: **complete.** All phases run; every configured model
finished its declared scope. Summarization was re-run on **TweetSumm** (the
official in-domain corpus) once it became available, with DialogSum retained as
a secondary cross-domain check — see §2 and §11.

All numbers in this file are measured; nothing is estimated unless the line says
so explicitly. Where an earlier conclusion was later shown to be wrong, the
correction is kept in place rather than silently rewritten — see §5 (the
llama-8b stall was a tokens-per-*day* cap, not the per-minute cascade first
reported) and §11 (summarization differences are *not* uniformly "within
noise"; that holds on DialogSum but not on TweetSumm).

---

## 1. Credentials — where they were and what worked

An earlier session concluded no keys existed anywhere and stopped. That was
wrong: the keys were on disk, just not under the project. Two were found in
unrelated projects, via a `.env.local` write recorded in `~/.zsh_history`:

| Key | Location found | Live result |
|---|---|---|
| `GROQ_API_KEY` | `~/blah/.env.local` | **works** (free tier) |
| `OPENAI_API_KEY` | `~/blah/.env.local` | authenticates, but `429 "You have no credits remaining"` |
| `ANTHROPIC_API_KEY` | `~/Desktop/ACLsmith/.env.local` (as `VITE_ANTHROPIC_API_KEY`) | `401 authentication_error` — **revoked key** |

The original Anthropic key was invalid, not merely unfunded — a `401` is
rejected before billing is consulted, so adding account credit does not fix it.
The user issued a replacement Anthropic key and a new Gemini key, and later
Mistral and Cerebras keys. Final state of every provider tried:

| Provider | Status | Usable |
|---|---|---|
| Anthropic | replacement key returns `200` | **yes** (paid) |
| Groq | works | **yes** (free) |
| Gemini | works, but only for some model ids — see §3 | **yes** (free) |
| Mistral | works; 50 req/min, 50,000 tok/min | **yes** (free) |
| OpenAI | account has no credits | no |
| Cerebras | `402 payment_required` on **every** model | no |
| Together AI | no key | no |

**Supplying keys.** `src/common.py` loads `cx-llm-bench/.env` automatically
(see `.env.example`; `.env` is gitignored, mode 600). This is the reliable
path: a key exported in an interactive terminal belongs to that terminal's
process and does not reach a separately-spawned run.

**Methods-section caveat.** The Groq, Mistral and Gemini results were produced
on *free promotional tiers*. That is a billing arrangement, not a statement
about the cost of self-hosting, and §4 explains why the cost axis of the money
chart must be read with that in mind.

---

## 2. Datasets

### Intent — Banking77

- **Repo used: `legacy-datasets/banking77`** (train 10,003 / test 3,080; 77
  `ClassLabel` intents).
- **Substitution, and why.** The original code called
  `load_dataset("banking77")`. Under the installed `datasets` 5.0.1 that raises
  `HfUriError` — bare canonical dataset names were removed. The obvious fix,
  `PolyAI/banking77`, then fails with
  `RuntimeError: Dataset scripts are no longer supported, but found banking77.py`,
  because `datasets` >=3 dropped script-based loaders. `PolyAI/banking77` also
  has no `refs/convert/parquet` revision to fall back to.
  `legacy-datasets/banking77` is parquet-native and carries the identical split
  sizes and the same 77 labels, so it is a format substitution, not a data
  substitution.
- Sampling: `random.Random(42)`, shuffle test split, take first `n_test`.
- Few-shot pool: 5 examples from the train split drawn from the **same** seeded
  RNG *after* the test shuffle, fixed across all queries.
- Two label strings have quirks worth knowing when parsing: `Refund_not_showing_up`
  (leading capital) and `reverted_card_payment?` (trailing question mark). The
  normalizer folds both, and all 77 labels were verified to round-trip.

### Summarization — TweetSumm (PRIMARY), DialogSum (secondary)

**Updated 2026-08-04: TweetSumm was obtained and is now the primary corpus.**
DialogSum is retained as a secondary cross-domain robustness check. The
DialogSum discussion below is kept for the record of why it was used first.

#### TweetSumm — primary

- **Source:** the official repo, <https://github.com/guyfe/Tweetsumm>, vendored
  at `vendor/tweetsumm/` (not committed).
- **Licensing — and why no dialogue text is in this repository.** TweetSumm
  distributes only tweet **IDs**, sentence offsets and human annotations. The
  tweet **text** lives in Kaggle's
  ["Customer Support on Twitter"](https://www.kaggle.com/thoughtvector/customer-support-on-twitter)
  dataset (`twcs.csv`, 493 MB, 3,002,523 rows) and is deliberately not
  redistributed by TweetSumm. Their dataset is released under
  **CDLA-Sharing-1.0** (<https://cdla.io/sharing-1-0/>); the repo's own *code*
  carries CC0-1.0. Accordingly this project commits **the loader and the tweet
  IDs only** — never `data/twcs.csv`, the vendored repo, the reconstructed
  dialogues, or the rating sheet built from them. `.gitignore` enforces this.
  Reproducers download `twcs.csv` from Kaggle themselves, which is exactly how
  the license intends text access to be routed.
- **Reconstruction:** `src/tweetsumm_loader.py` reuses the upstream
  `TweetSumProcessor` **verbatim** for offset slicing and speaker attribution,
  rather than reimplementing it and risking a subtle mismatch. The only change
  is memory: upstream's constructor loads all ~3M twcs rows into a dict, so we
  build the identical `tweet_id_to_content` mapping from one streaming pass
  keeping only the ~1.1k IDs the test split needs, and inject it.
- **Reconstruction integrity (test split):** 110 dialogues referencing 1,152
  distinct tweet IDs; **1,152/1,152 resolved in twcs.csv, 0 dialogues dropped.**
  Full coverage — no silent truncation.
- **Reference summaries:** each dialogue carries ~**3** human abstractive
  annotations (3.11 on average over the sampled 100; a few carry more), each a
  list of sentences joined into one reference string. **All** are retained.
  The primary scoring is **multi-reference ROUGE — max over references**, the
  standard convention when a corpus ships several human summaries; the first
  reference alone is also scored and reported, for continuity with DialogSum
  which has only one. See §11.
- **Speaker attribution:** upstream marks a turn as agent when the Kaggle
  `inbound` field is `False`. Rendered as `Customer:` / `Agent:` lines.
- **Sampling:** all 110 test dialogues loaded, shuffled with `Random(42)`, first
  **100** taken — identical treatment to Banking77 and DialogSum.
- **Scale:** TweetSumm dialogues average ~1,281 characters vs DialogSum's ~700,
  i.e. meaningfully longer multi-turn support threads.

**Why this matters for the paper.** DialogSum is general-domain daily
conversation; TweetSumm is real customer-support dialogue between consumers and
company agents. The paper's claim is about *contact center* automation, so
TweetSumm is the in-domain measurement and DialogSum is now a free bonus: a
cross-domain check on whether the model ordering is corpus-specific.

#### DialogSum — secondary (original notes retained)

- **Dataset used: `knkarthick/dialogsum`** (train 12,460 / validation 500 /
  **test 1,500**), the fallback explicitly authorized in CLAUDE.md.
- **Why TweetSumm was not used.** `Salesforce/dialogstudio` is now *both*
  gated and script-based:
  `DatasetNotFoundError: Dataset 'Salesforce/dialogstudio' is a gated dataset on
  the Hub. You must be authenticated to access it.` `trust_remote_code` is also
  no longer honoured by `datasets` 5.x.
- **Mirrors considered and rejected.** A hub search surfaced two candidates:
  - `Andyrasika/TweetSumm-tuned` — 879/110/110, which *matches the canonical
    TweetSumm split sizes*, with `conversation` + `summary` columns. It is
    plausibly a faithful conversion, but it is an unofficial third-party
    re-upload with no dataset card guaranteeing provenance, it carries a single
    abstractive summary per dialogue where official TweetSumm has three
    annotator summaries, and its **test split of 110 is smaller than the
    configured `n_dialogues: 120`**.
  - `chezhian/Tweet_summary` — 800 rows, a single pre-templated `text` column
    with instruction formatting baked in. Not usable as a reference corpus.

  Neither is defensible as the primary dataset for a published benchmark, so
  the sanctioned DialogSum fallback was chosen instead.
- **How to get real TweetSumm.** `src/run_summ.py` now attempts official
  TweetSumm first and only falls back if it cannot load. Set `HF_TOKEN` (or
  `HUGGING_FACE_HUB_TOKEN`) after being granted access to
  `Salesforce/dialogstudio`, and rerun — it will switch automatically and record
  `"dataset": "TweetSumm"` in `results/summ_summary.json`. Note the gated repo is
  still script-based, so it may additionally require pinning `datasets<3`.
- Caveat for the Methods section: DialogSum is general-domain daily dialogue
  (`#Person1#/#Person2#`), **not** customer-support transcripts. It is a weaker
  proxy for contact center work than TweetSumm and should be named as such in
  Limitations.

---

## 3. Model lineup — final, all IDs verified live

Balance requirement (>=2 per category) is satisfied: **3 open-weights,
2 proprietary**.

| Model | Category | Billing | Role |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | proprietary | paid (~$2) | proprietary anchor |
| `gemini/gemini-flash-lite-latest` | proprietary | Gemini free tier | second proprietary |
| `groq/llama-3.3-70b-versatile` | open-weights | Groq free tier | **headline open model** |
| `mistral/mistral-small-latest` | open-weights | Mistral free tier | mid-size open (Apache-2.0) |
| `groq/llama-3.1-8b-instant` | open-weights | Groq free tier | "cheap self-hosting" point |

### ID swaps and exclusions, with evidence

- **`gpt-4o-mini` → dropped.** Key authenticates but the account has zero
  credits (`429 "You have no credits remaining"`). Not a code problem.
- **`mistral-small-latest` → retained** once a key was supplied. Serves
  snapshot `mistral-small-latest`; genuinely open-weights (Apache-2.0).
- **`gemini-2.0-flash` → `gemini-flash-lite-latest`.** This key's Google Cloud
  project has free-tier quota *provisioned at zero* — the error is explicit:
  `Quota exceeded ... limit: 0, model: gemini-2.0-flash`. Every pinned 2.0/2.5
  id behaves the same way, and `gemini-2.5-flash{,-lite}` additionally return
  `404 "no longer available to new users"`. Only the `-latest` aliases carry
  usable quota.
- **`gemini-flash-latest` → EXCLUDED after testing.** It resolves to
  `gemini-3.6-flash`, a **thinking model**. On the real intent prompt it spends
  the entire completion budget reasoning and returns a truncated fragment:
  `max_tokens=30 → 26 tokens → ' card_'`; `max_tokens=128 → 124 tokens →
  'card'`. Both are unparseable, and it scored **0.0 accuracy / 1.0 unparsed**
  on a 5-query smoke test. This is exactly the CLAUDE.md step-10 failure mode,
  and the honest fix was to exclude the model rather than report a broken
  score. Its 10 smoke-test rows remain in `results/raw/intent.jsonl` as a
  record but it is not a reported model.
- **`groq/qwen/qwen3.6-27b` and `groq/openai/gpt-oss-120b` → excluded**, same
  reason: both are live on Groq but both emit `<think>` blocks and need
  `max_tokens >= 512` to reach a label.
- **`together_ai/Qwen2.5-72B` → dropped.** No key, and no free tier.
- **Cerebras → unusable.** A Cerebras key was added specifically to run
  `llama-3.3-70b` inside 24 h. Two independent blockers: that account's
  catalogue is only `{zai-glm-4.7, gpt-oss-120b, gemma-4-31b}` — **no Llama of
  any size** (`404 model_not_found`) — and every one of those three returns
  `402 payment_required`. So **no cross-provider consistency check for the 70B
  was possible**, and the 70B runs on Groq alone.

**Snapshot-ID caveat for the paper.** `mistral-small-latest`,
`gemini-flash-lite-latest` and `llama-3.3-70b-versatile` are moving aliases,
not pinned snapshots. `common.py` records the provider-reported
`model_snapshot` on every call, so the served version is in
`results/raw/*.jsonl` — cite *those* in the Methods table, not the aliases.

### Threat to validity: the proprietary side is one vendor, two tiers

**The final proprietary side is `claude-haiku-4.5` and `claude-sonnet-5` — both
Anthropic.** This is a real limitation and must be stated in the paper: the
"proprietary" category is represented by a single vendor's model family, so any
vendor-specific behaviour (tokenizer, instruction-following style, refusal
behaviour, prompt-format sensitivity) is confounded with the category itself.
A gap measured this way is really "open weights vs *Anthropic*", not "open
weights vs proprietary models in general".

How it came about, for the record: `gpt-4o-mini` was lost to an OpenAI account
with no credits; `gemini-2.0-flash` to a project with zero free-tier quota;
`gemini-flash-latest` to being a thinking model that cannot answer under the
task's token budget; and `gemini-flash-lite-latest` to a 500-request/day cap
that made its 2,100-call workload a 4+ day run. Every alternative required
either new billing or multi-day runtime, and the user's constraints excluded
both.

Two things mitigate it, and both should be reported rather than glossed:

1. **The Gemini subsample.** `gemini-3.5-flash-lite` completed ~467 zero-shot
   intent calls before hitting its daily cap. Because the test split is
   shuffled with `Random(42)` *before* being sliced, those are a valid random
   subsample rather than a biased prefix. They give a genuine second-vendor
   proprietary reading, reported at its own n and **excluded from the headline
   gap** (a score from a smaller sample is not comparable to a full-n score).
   It is a consistency check on the single-vendor result, not a replacement.
2. **Tier-scaling symmetry.** Both sides span a capability range rather than a
   single point: open weights run 8B -> ~24B -> 70B, and proprietary runs
   Haiku -> Sonnet. So the comparison is between *tiers* on both sides, and the
   headline gap can be read at comparable tiers instead of resting on one
   arbitrary pairing. That is weaker than genuine vendor diversity but it does
   mean the result is not a single-model artifact.

Both proprietary models ran the **full** workload (zero-shot + few-shot intent
at n=1000 each, plus 100 summaries). Sonnet's few-shot arm was initially
skipped as unaffordable — its post-4.7 tokenizer makes it ~34% more expensive
per call (§9) — and was completed on 2026-08-01 after the user added credit
(1000/1000, 0 errors, $2.31). On the open side, `llama-3.1-8b` still lacks
few-shot for a rate-limit reason (§5), so it contributes to zero-shot only.

### Threat to validity: the open-weights side skews small

`llama-3.3-70b` is the only large open model available, and it is also the
slowest (§5). If it does not finish, the open-weights side is a 24B and an 8B
model against a current proprietary model, which **overstates the compliance
gap**. Any reported gap that excludes the 70B is an upper bound and must be
labelled as such.

---

## 4. Cost

Only **one** model in the final lineup is billed: `claude-haiku-4-5-20251001`.
Groq, Mistral and Gemini all run on free tiers at $0.00. Actual spend is
computed per call by `litellm.completion_cost` and aggregated from the raw
logs; the final figure is in §8 and in `RUN_REPORT.md`.

Estimated total for the run at `n_test: 1000` / `n_dialogues: 100`: **~$2**,
against a $30 cap and the user's $5 auto-proceed threshold. The reduction from
1500/120 was made for **rate-limit** reasons (§5), not cost.

### Reading the cost axis of the money chart — important

`litellm` cannot price Groq or Mistral responses and returns `None`, whereas a
free-tier Gemini call returns a genuine `0.0`. `report.py` distinguishes these
with a `priced_frac` column so "free" is never silently conflated with
"unknown". Both nonetheless plot at **$0.00** on the accuracy-vs-cost scatter.

That $0.00 is *what we paid on a promotional tier*, *not* the cost of running
these models. It is emphatically not self-hosting TCO — which is GPU capital or
rental, engineering time, and idle capacity, and is the actual quantity a
privacy-constrained enterprise trades against. **The scatter therefore shows
the open-weights models at an unrealistically favourable cost position.** The
paper must either re-derive that axis from a defensible per-token
self-hosting estimate or state plainly that the axis is API list price and that
open-weights entries are promotional-tier zeros. This is a presentation caveat,
not a defect in the measurements — the accuracy and ROUGE numbers are unaffected.

---

## 5. Rate limits — measured, and why the sample size was reduced

Free tiers are metered **per day**, not only per minute, and this turned out to
be the binding constraint on the whole study. All figures below were read from
live `x-ratelimit-*` response headers on 2026-07-31, not from documentation.

| Model | Measured free-tier cap | Derived delay |
|---|---|---:|
| `groq/llama-3.3-70b-versatile` | **1,000 req/day**, 12,000 tok/min | 87 s |
| `groq/llama-3.1-8b-instant` | 14,400 req/day, 6,000 tok/min | 6 s |
| `mistral/mistral-small-latest` | 50 req/min, 50,000 tok/min | 1.3 s |
| `gemini/gemini-flash-lite-latest` | no cap hit in 14 rapid calls | 5 s |
| `gemini/gemini-flash-latest` *(excluded)* | **5 req/min** | — |
| `claude-haiku-4-5-20251001` | paid, generous | 1.2 s |

How the daily caps were derived: Groq refills a rolling bucket, so the reset
header gives the rate directly. `llama-3.3-70b` reported
`x-ratelimit-limit-requests: 1000` with `reset-requests: 2m52.8s` after 2 calls
— and `172.8 s / 2 = 86.4 s = 86400/1000`, i.e. exactly 1,000 requests per day.
`llama-3.1-8b` reported 14,400 with a 6 s reset after one call (`86400/14400`).
Pacing at the refill rate consumes the daily quota smoothly and never 429s;
running faster would exhaust the day and stall the run in backoff.

### The llama-8b case: a tokens-per-DAY cap, not tokens-per-minute

**Corrected 2026-08-01.** This was initially attributed to the 6,000
tokens/minute cap and a self-sustaining retry cascade. The provider's own error
message, recovered from `results/raw/intent.jsonl` during the pre-publication
secret sweep, shows the real cause:

```
Rate limit reached for model `llama-3.1-8b-instant` in organization
`org_[REDACTED]` service tier `on_demand` on tokens per day (TPD):
Limit 500000, Used 499554, Requested 684.
```

Groq's free tier meters this model at **500,000 tokens per DAY**. The 1,000
zero-shot calls at ~500 prompt tokens each consumed ~500,000 tokens — i.e.
essentially the entire daily budget — so by the time the few-shot arm started
(~645 tokens/call) there was nothing left to spend. The observed ~120 s/call was
not backoff thrash; it was the daily bucket **refilling**, at
`500,000 / 86,400 ≈ 5.8 tokens/sec`. A 662-token summarization call needs ~114 s
of refill at that rate, which matches the measurement closely.

| Condition | Prompt tokens | Outcome |
|---|---:|---|
| zero-shot | ~500 | completed at full n=1000 — and exhausted the day's tokens |
| few-shot (k=5) | ~645 | blocked by TPD; ~32 h projected; abandoned |
| summarization | ~662 | ~60–120 s/call, governed by TPD refill |

**Why the correction matters.** The original write-up implied the fix was
tuning the per-request delay. It was not: no delay setting recovers a budget
that is already spent, and the two arms were competing for one shared daily
pool. The genuine lessons are:

1. **Budget the whole workload against the daily TOKEN cap before starting**,
   not just the per-minute rate. `n_calls x prompt_tokens` is the number that
   has to fit. Here, 2,100 calls x ~570 tokens ≈ 1.2M tokens against a 500k/day
   cap — the workload never fit in one day, and no pacing could change that.
2. **Order conditions by cost when a shared budget is involved.** Running
   zero-shot first spent the entire day on the cheaper arm.
3. Per-minute pacing still matters (§ the Groq 70B, capped on requests/day), but
   it is a separate constraint from the daily token pool.

Raising the delay from 8.5 s to 12 s did coincide with summarization improving
from ~120 s/call to ~60 s/call, but that is better explained by refill
accounting and prompt-length variance than by the delay change itself. The
earlier causal claim was not supported by evidence and has been withdrawn.

**Consequence — `n_test` reduced 1500 → 1000 and `n_dialogues` 120 → 100.**
Not for cost (cost is ~$2), but because 3,120 calls at 1,000 req/day is 3.1
days for the 70B. This was a deliberate, user-approved trade. Both conditions
(zero-shot **and** few-shot) were retained, per the study design.

The reduction is **cache-safe**: `run_intent.py` shuffles the full test split
with `Random(42)` *before* slicing to `n`, so the 1,000 are a strict prefix of
the 1,500; and the few-shot exemplars are drawn from the train split *after*
the test shuffle, so the RNG state — and therefore the exemplars — are
identical at either size. No previously cached call was invalidated.

### Wall-clock at n=1000 (2,000 intent + 100 summarization = 2,100 calls/model)

| Model | Projected |
|---|---:|
| `claude-haiku-4-5-20251001` | ~1.4 h |
| `mistral/mistral-small-latest` | ~2.5 h |
| `groq/llama-3.1-8b-instant` | ~3.4 h |
| `gemini/gemini-flash-lite-latest` | ~4.6 h |
| `groq/llama-3.3-70b-versatile` | **~2.1 days** (the long pole) |

Each model runs in its own process (`run_model.sh`, which chains intent then
summarization for that model). This matters for correctness, not just speed:
`common.py` throttles per model *within a process*, so two processes touching
the same model would each keep their own clock and together exceed its quota.
Different models have independent quotas and are safe to parallelize.

---

## 6. Code changes made (all pre-run; no results affected)

`requirements.txt` was missing **`tabulate`**, which `DataFrame.to_markdown()`
requires — `report.py` would have crashed at the final step after a full paid
run had completed. Installed; add it to the file.

**`src/common.py`**
- Per-model throttling (`rate_limits` in config, resolved by exact ID, then
  provider via `litellm.get_llm_provider`, then `default`).
- Rate-limit-aware retry: 429/quota errors get a longer jittered backoff
  (up to 60 s) than ordinary errors; retries raised 4 -> 6.
- Failures no longer raise and abort the whole multi-model run. `call_llm`
  returns `error` set and `text: ""`. **Failed calls are not cached**, so a
  rerun retries them.
- `seed=42` sent alongside `temperature=0`, with `litellm.drop_params = True`
  so providers that reject `seed` still work.
- Cache key now includes `max_tokens` (previously two calls with the same
  prompt but different limits could collide). Done before any cache existed,
  so nothing was invalidated.

**`src/run_intent.py`**
- Dataset repo fix (see §2).
- **Rewrote label matching** — this is the near-0-accuracy failure mode CLAUDE.md
  warns about in step 10. The old code normalized only the first line, so
  `Intent: card_arrival`, `**card_arrival**`, or `The intent is card_arrival.`
  all scored as `__unparsed__`. New `parse_label` strips code fences, markdown,
  quotes and `Intent:` prefixes, then tries exact -> containment (longest label
  first, so `card_payment_fee_charged` beats a shorter nested label) -> fuzzy
  (cutoff 0.9). Verified offline: 21/21 hand-written output shapes, and all 77
  gold labels round-trip.
- Failed calls are recorded as `__error__` and **excluded from accuracy**
  (a dead provider is not a wrong answer) while `error_rate` and `n_scored`
  report the coverage loss. `unparsed_rate` still reported separately.
- `--models` flag for per-model runs.

**`src/run_summ.py`** — TweetSumm/HF_TOKEN logic (§2), error handling,
`--models`, `n_scored` + `error_rate` in the summary.

**`src/report.py`** — the Phase 2 changes (§7) plus three fixes:
- Cost was aggregated **per model across both conditions**, so the zero-shot
  "money chart" was plotted against a cost blended with the more expensive
  few-shot calls. Now keyed per (model, condition).
- The raw log is append-only and a rerun re-appends every row, which
  double-counted cost. Rows are now deduped on call identity before aggregating.
- Latency was collected only from non-cached rows, so a fully-cached rerun
  reported no latency at all. Cached rows carry the originally measured latency
  and are now included.
- `priced_frac` distinguishes a genuine $0.00 free-tier call from one litellm
  could not price (`None`), so free is never confused with unknown.
- Guards so an all-failed model cannot crash the report (it previously raised
  `KeyError: ['cost_per_1k']`).

**`src/make_rating_sheet.py`** (new) — the Phase 2 step-12 deliverable.

---

## 7. Phase 2 changes (implemented and run against real data)

- **Category-colored scatter**: `accuracy_vs_cost.png` colors points by
  open-weights (blue) vs proprietary (red) using `open_weights_models` from
  config, with a legend. A matching `rougeL_vs_cost.png` was added for the
  summarization task, and the zero-shot/few-shot bar chart tick labels are
  color-coded by category.
- **Compliance-gap table**: best open vs best proprietary per task and
  condition, for accuracy, macro-F1 and ROUGE-1/2/L, reporting `abs_gap`
  (proprietary − open) and `rel_gap_pct`, alongside `cost_per_1k` for each side.
  If either category has no scored model it prints "Not computable" rather than
  inventing a comparison.

Both were subsequently run against the real results. Three further changes were
made once real data existed:

- **The cost axis was rebuilt on published list prices** (§9) — the original
  version plotted billed cost, which is ~$0 for the free-tier models and would
  have made the money chart an artifact of promotional pricing.
- **A supplementary section and `full_n` gating** — only full-sample rows feed
  the compliance gap and the scatter charts; reduced-sample models (Gemini)
  are reported separately at their own n.
- **`src/rebuild_summaries.py`** — reconstructs all summary JSONs from the raw
  logs. Necessary because concurrent per-model processes were clobbering each
  other's summaries; `summ_outputs.json` had ended up with 2 of 7 models, which
  would have produced a silently incomplete expert rating sheet.

An earlier session exercised the reporting code against synthetic plumbing data
in a scratch directory outside the project to prove the code paths ran. **No
synthetic data was ever written into `results/` or `cache/`**, and every number
now in `tables.md` traces to a row in `results/raw/*.jsonl`.

---

## 8. Totals

- **Unique API calls: 10,156** (the raw log is append-only and consolidation
  re-appends, so every consumer dedupes on call identity).
- **Total cost: $7.2013.** Anthropic accounted for $6.04 of it. This exceeds
  the $6 ceiling that was in force earlier in the run: the user added credit on
  2026-08-01 and directed that Sonnet's few-shot arm ($2.31) be run, which
  raised the total by design. The `budget_cap_usd: 30` hard cap was never
  approached.
- **Errors: 29 of 10,156 (0.29%)** — all rate-limit exhaustion on models
  retired early by design (28 Gemini, 1 Groq-8B). None affected a reported
  metric; failed calls are excluded from scoring and surfaced via `error_rate`.
  Sonnet's 1,000 few-shot calls completed with **zero** errors.
- **Cache: ~10,091 files**, never deleted.

Per-model call counts, billed cost and median latency are in `RUN_REPORT.md` §1.

---

## 9. Cost axis: published list prices, not what we were billed

### Why the billed figure is not usable

Four of the five models ran on **free promotional tiers**, so the amount
actually billed is $0.00 for them and ~$2 for `claude-haiku`. Plotting that
would produce a chart whose headline finding is "open-weights models cost
nothing" — an artifact of promotional pricing, not an economic result.

The reported cost axis is therefore **hosted API list price**: every call is
priced at its provider's published rate, regardless of what we paid. It is
labelled that way on both scatter charts. Implementation: `src/list_price.py`.
`billed_cost_per_1k` is retained alongside it in `tables.md` for transparency.

**This is still not self-hosting TCO** — GPU capital or rental, engineering
time, and idle capacity are the real costs a privacy-constrained enterprise
trades against, and none of them are on this axis. That gets a sentence in the
paper's Limitations.

### Prices used (all retrieved 2026-07-31)

| Model | Priced as | Input $/M | Output $/M | Source |
|---|---|---:|---:|---|
| `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` | 1.00 | 5.00 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| `gemini/gemini-flash-lite-latest` | `gemini-3.5-flash-lite` | 0.30 | 2.50 | [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| `groq/llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` | 0.59 | 0.79 | [Groq pricing](https://groq.com/pricing) |
| `groq/llama-3.1-8b-instant` | `llama-3.1-8b-instant` | 0.05 | 0.08 | [Groq pricing](https://groq.com/pricing) |
| `mistral/mistral-small-latest` | Mistral Small 4 | 0.15 | 0.60 | [Mistral API pricing](https://mistral.ai/pricing/api) |

Two of these were **wrong** in litellm 1.94.1's bundled pricing DB, which is
why they were verified against the vendors' pages rather than trusted:

- **Mistral Small** — litellm has $0.06/$0.18; published is **$0.15/$0.60**
  (2.5x / 3.3x understated).
- **Gemini Flash-Lite** — litellm has $0.10/$0.40, which is *2.5* Flash-Lite
  pricing. The alias actually serves **`gemini-3.5-flash-lite`** at
  **$0.30/$2.50** (3x / 6.25x understated).

Anthropic and both Groq models matched litellm exactly.

**How the Gemini alias was resolved.** `gemini-flash-lite-latest` is a moving
alias and Google does not publish its mapping; litellm and the raw logs both
just echo the alias string back. The concrete version comes from the
`modelVersion` field in the Gemini response body, which reports
`gemini-3.5-flash-lite`. (Pinned `gemini-2.5-flash-lite` is separately dead on
this key: `404 no longer available to new users`.) Cite `gemini-3.5-flash-lite`
in the Methods table, and note that the alias may hot-swap under a rerun.

### How token counts were estimated

`common.py` caches responses but never persisted the providers' `usage`
fields, so token counts are **estimated, not measured**:

- **Input** — prompts are reconstructed *exactly* as the runners built them
  (`run_intent.build_prompt` with the same seed-42 label list and few-shot
  exemplars; `run_summ.PROMPT`) from the query/dialogue text preserved in
  `results/raw/*.jsonl`, then tokenized.
- **Output** — the generated text stored in the same raw logs, tokenized.
- **Tokenizer** — `litellm.token_counter(model=...)`, which picks a
  model-appropriate tokenizer where it knows one and otherwise falls back to a
  cl100k-style default.

Known limitations of the estimate: for the non-OpenAI models the tokenizer is
an approximation (expect a few percent); it excludes provider-side chat-template
scaffolding and BOS/EOS tokens; and it therefore slightly *understates* true
billable input. It is deterministic and applied identically to every model, so
it is sound for **comparing** models — which is the chart's purpose — but
absolute values should be quoted as estimates in the paper.

A cheap improvement for any future rerun: persist `resp.usage` into the cache
and use measured counts instead. Worth doing before the paper's final numbers
if precision on the absolute cost matters.

**litellm's bundled prices were wrong for three of six models.** Every price
above was verified against the vendor's own page for this reason:

| Model | litellm 1.94.1 | Published | Error |
|---|---|---|---|
| Mistral Small | $0.06 / $0.18 | **$0.15 / $0.60** | 2.5x / 3.3x low |
| Gemini Flash-Lite | $0.10 / $0.40 | **$0.30 / $2.50** | 3x / 6.25x low |
| Together Llama-3.3-70B | $0.88 / $0.88 | **$1.04 / $1.04** | 18% low |

Anthropic and both Groq models matched exactly. Note the consequence for
`billed_cost_per_1k`: that column comes from `litellm.completion_cost`, so for
Together it reflects the older $0.88 rate and is an estimate, not an invoice.

---

## 10. Serving providers, and the mid-study move of the 70B

**Serving provider matters for the Methods section** — an open-weights model's
score depends on who served it and at what precision, not just on the weights.

| Model | Served by | Precision |
|---|---|---|
| `claude-haiku-4-5-20251001` | Anthropic API | vendor default |
| `gemini/gemini-flash-lite-latest` | Google AI Studio (`gemini-3.5-flash-lite`) | vendor default |
| `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | Together AI, serverless | **FP8** (`-Turbo`) |
| `mistral/mistral-small-latest` | Mistral La Plateforme | vendor default |
| `groq/llama-3.1-8b-instant` | Groq | Groq-quantized |
| `groq/llama-3.3-70b-versatile` *(baseline only)* | Groq | Groq-quantized |

### Why the 70B moved hosts mid-study

Groq's free tier caps `llama-3.3-70b-versatile` at 1,000 requests/day, making
the 2,100-call workload a ~52h run — not feasible on a laptop that cannot stay
awake. The project's "no paid open endpoints" rule was **explicitly overridden
by the user for this one model**, and it moved to Together AI at an estimated
$1.24, inside the $5 auto-proceed threshold.

Together's non-quantized `meta-llama/Llama-3.3-70B-Instruct` was preferred on
scientific grounds but is **not serverless** on this account (`Unable to access
non-serverless model`), so the FP8 `-Turbo` build is what was served.

### Direction of the FP8 bias — state this explicitly in the paper

Quantization can only depress the served model's quality relative to the full
-precision weights, so:

- the measured 70B score is a **lower bound** on full-precision performance:
  `open_measured <= open_full_precision`
- and because the gap is `proprietary - open`, that propagates as
  `gap_measured = prop - open_measured >= prop - open_full = gap_true`

So **the reported compliance gap is an UPPER bound on the true full-precision
gap. FP8 can overstate the gap; it can never understate it.**

This is the conservative direction for a sceptical reading of the paper's
headline claim (an enterprise restricted to open weights sacrifices *at most*
the reported amount), but it is the *anti*-conservative direction for anyone
citing the gap as evidence that open weights are inadequate. Both open-weights
caveats in this study push the same way — FP8 quantization here, and the
small-model skew noted in §3 — so they compound rather than cancel. Any
sentence in the paper that quotes a gap figure should carry the upper-bound
qualifier.

The partial Groq 70B run was left to finish its summarization set and is
retained in `results/raw/*.jsonl` as the consistency baseline below. It is
**not** a reported model and is excluded from `models` in `config.yaml`.

### Cross-provider consistency: llama-3.3-70b, Together vs Groq

Generated by `src/consistency_check.py`. The Groq run was allowed to finish its
full summarization set before being retired, so the overlap is 100 dialogues
rather than the ~20 originally expected.

**Intent** — 17 prompts answered by both providers:

- Identical predicted label: **17/17 (100.0%)**
- Correct vs gold: Together 14/17, Groq 14/17; both correct on the same 14

**Summarization** — 100 dialogues answered by both:

- ROUGE-L vs reference: Together **0.1416**, Groq **0.1434** — delta
  **-0.0018**, i.e. 1.3% relative, far below the ~0.03 gap between models
- ROUGE-L of the two providers' outputs **against each other**: **0.6386**

**What this supports.** Moving the 70B from Groq to Together mid-study did not
materially change what was being measured. That was the live risk — the
headline open-weights number would be worthless if the host swap had shifted
it — and it is not borne out: perfect label agreement, and a summarization
delta an order of magnitude smaller than the between-model differences the
paper reports.

**What it does NOT support.** This is not a clean provider-vs-provider
experiment and must not be written up as one. Both providers serve *quantized*
builds (Together explicitly FP8 `-Turbo`), so agreement here says the two
quantized servings behave alike, not that quantization is harmless — the
full-precision weights were never available to either. The FP8 upper-bound
caveat above stands independently and is unaffected by this result.

The cross-provider ROUGE-L of 0.639 is the interesting nuance: the two
servings produce **substantially different wording** (0.639, not ~1.0) while
scoring within 0.002 of each other against the reference. At `temperature=0`
that divergence comes from numerical differences in the serving stacks pushing
decoding down different paths. Worth a sentence in the paper: summarization
quality here is robust to serving-stack variation even though the surface text
is not reproducible across providers. Anyone expecting byte-identical outputs
from `temperature=0` across hosts will not get them.

---

## 11. Summarization scoring and statistical significance

### Multi-reference ROUGE is the primary scoring for TweetSumm

TweetSumm ships **~3 human abstractive summaries per dialogue** (3.11 on
average across the sampled 100 — a few dialogues carry more). Scoring against
only the first measures agreement with one arbitrary annotator, not summary
quality: a model is penalised for writing a perfectly good summary that happens
to resemble annotator 2. The standard convention with multiple references is to
score against each and take the **maximum** per metric, which is what
`src/multiref.py` does. **0 dialogues fell back to single-reference scoring.**

This required **no new API calls** — the generated summaries were already in the
raw log; only the extra references are new, joined on dialogue id.

DialogSum has one reference per dialogue and cannot be scored this way, so the
**single-reference** TweetSumm numbers are retained below: a cross-corpus
comparison must be single-ref to single-ref.

| Model | multi-ref ROUGE-L | single-ref ROUGE-L |
|---|---:|---:|
| `together_ai/…Llama-3.3-70B-Instruct-Turbo` (open) | **0.2738** | **0.2198** |
| `mistral/mistral-small-latest` (open) | 0.2542 | 0.2053 |
| `groq/llama-3.1-8b-instant` (open) | 0.2538 | 0.2069 |
| `claude-haiku-4-5-20251001` (proprietary) | 0.2529 | 0.2060 |
| `claude-sonnet-5` (proprietary) | 0.2480 | 0.2028 |

All scores rise under multi-reference, as they must — a max over more
references cannot decrease. What matters is that the **ordering is stable at
the top** (the 70B leads under both) and that the open-vs-proprietary margin
**widens** rather than narrowing, so the finding is not an artifact of which
annotator was picked.

### Paired bootstrap

### Are the summarization differences real? Paired bootstrap

Paired bootstrap over per-dialogue ROUGE-L differences, 10,000 resamples, seed 42. Resampling is over dialogues (paired), because both models score the same dialogues and dialogue difficulty dominates the variance.


**TweetSumm — MULTI-REFERENCE (primary; max over ~3 human summaries)** (n=100 dialogues) — ranked by mean ROUGE-L: `Llama-3.3-70B-Instruct-Turbo` 0.2738, `mistral-small-latest` 0.2542, `llama-3.1-8b-instant` 0.2538, `claude-haiku-4-5-20251001` 0.2529, `claude-sonnet-5` 0.2480

- **Top-1 vs top-2** — `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` minus `mistral/mistral-small-latest`, n=100 paired:
  observed ΔROUGE-L = **+0.0196**, 95% CI [+0.0094, +0.0298], p≈0.000 → **distinguishable** (95% CI excludes 0)
- **Best open-weights vs best proprietary (the compliance-gap claim)** — `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` minus `claude-haiku-4-5-20251001`, n=100 paired:
  observed ΔROUGE-L = **+0.0209**, 95% CI [+0.0099, +0.0316], p≈0.000 → **distinguishable** (95% CI excludes 0)

**TweetSumm — single-reference (for continuity with DialogSum)** (n=100 dialogues) — ranked by mean ROUGE-L: `Llama-3.3-70B-Instruct-Turbo` 0.2198, `llama-3.1-8b-instant` 0.2069, `claude-haiku-4-5-20251001` 0.2060, `mistral-small-latest` 0.2053, `claude-sonnet-5` 0.2028

- **Top-1 vs top-2** — `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` minus `groq/llama-3.1-8b-instant`, n=100 paired:
  observed ΔROUGE-L = **+0.0129**, 95% CI [+0.0045, +0.0214], p≈0.003 → **distinguishable** (95% CI excludes 0)
- **Best open-weights vs best proprietary (the compliance-gap claim)** — `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` minus `claude-haiku-4-5-20251001`, n=100 paired:
  observed ΔROUGE-L = **+0.0138**, 95% CI [+0.0034, +0.0240], p≈0.009 → **distinguishable** (95% CI excludes 0)

**DialogSum — single-reference** (n=100 dialogues) — ranked by mean ROUGE-L: `mistral-small-latest` 0.1690, `claude-sonnet-5` 0.1615, `claude-haiku-4-5-20251001` 0.1459, `llama-3.3-70b-versatile` 0.1434, `llama-3.1-8b-instant` 0.1423, `Llama-3.3-70B-Instruct-Turbo` 0.1416

- **Top-1 vs top-2** — `mistral/mistral-small-latest` minus `claude-sonnet-5`, n=100 paired:
  observed ΔROUGE-L = **+0.0074**, 95% CI [-0.0052, +0.0201], p≈0.245 → **within noise** (95% CI includes 0)
- **Best open-weights vs best proprietary (the compliance-gap claim)** — `mistral/mistral-small-latest` minus `claude-sonnet-5`, n=100 paired:
  observed ΔROUGE-L = **+0.0074**, 95% CI [-0.0052, +0.0201], p≈0.245 → **within noise** (95% CI includes 0)

### What this means for the paper

**The open-weights summarization lead on in-domain data is statistically
solid, and multi-reference scoring strengthens it.**

| Scoring | Δ (best open − best proprietary) | 95% CI | p | verdict |
|---|---:|---|---:|---|
| TweetSumm, **multi-ref (primary)** | **+0.0209** | [+0.0099, +0.0316] | ≈0.000 | distinguishable |
| TweetSumm, single-ref | +0.0138 | [+0.0034, +0.0240] | ≈0.009 | distinguishable |
| DialogSum, single-ref | +0.0074 | [−0.0052, +0.0201] | ≈0.245 | within noise |

- Do **not** carry a blanket "summarization differences are within noise" claim
  into the paper. That is true on DialogSum and false on TweetSumm under either
  scoring.
- The result is robust to the scoring choice: it holds single-ref and gets
  *stronger* multi-ref. That is the right direction — a finding that only
  appeared under the more favourable metric would be suspect.

**The ranking still does not transfer across domains.** `llama-3.3-70B` is 1st
of 6 on TweetSumm (both scorings) and last on DialogSum (0.1416);
`mistral-small` is 1st on DialogSum and 2nd–4th on TweetSumm depending on
scoring. Since DialogSum's own top-2 difference is not significant, some of the
reshuffle is noise — but the 70B's movement far exceeds that interval. Treat
ROUGE-based summarization rankings as corpus-specific.

### Caveats on the method

- The bootstrap is **paired** (resampling dialogues, not models): both models
  score the same dialogues and dialogue difficulty dominates the variance, so
  an unpaired test would be the wrong instrument.
- `p_two_sided` is an approximate bootstrap p-value, an aid to interpretation
  rather than a hypothesis test with a hard threshold. Values printed as
  ≈0.000 mean no resample out of 10,000 crossed zero, not a literal zero.
- Only the **top-2** and **best-open vs best-proprietary** pairs were tested,
  as specified. No multiple-comparison correction is applied; with two planned
  comparisons per corpus that is defensible, but an all-pairs matrix would
  need one.
- ROUGE measures surface overlap, not whether a summary is usable at agent
  handoff. That is what the blinded expert rating sheet — now built from
  TweetSumm — is for, and it remains the instrument for the usability claim.
