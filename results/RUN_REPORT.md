# RUN REPORT — CX-LLM-Bench

**Date:** 2026-07-31; Sonnet few-shot completed 2026-08-01; summarization re-run
on TweetSumm 2026-08-04/05
**Outcome:** **Complete.** All configured models finished their declared scope;
every acceptance-criteria artifact exists.
**Total actual cost: $7.6221. Total API calls: 10,656 unique.**

---

## 1. Cost and calls

| Model | Calls | Errors | Billed | Median latency |
|---|---:|---:|---:|---:|
| `claude-haiku-4-5-20251001` | 2,200 | 0 | $1.7171 | 0.699 s |
| `mistral/mistral-small-latest` | 2,200 | 0 | $0.0156 | 0.548 s |
| `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | 2,200 | 0 | $1.1619 | 0.611 s |
| `claude-sonnet-5` | 2,200 | 0 | $4.6931 | 1.278 s |
| `groq/llama-3.1-8b-instant` | 1,234 | 1 | $0.0000 | 0.304 s |
| `gemini/gemini-flash-lite-latest` | 495 | 28 | $0.0317 | 0.482 s |
| `groq/llama-3.3-70b-versatile` *(baseline)* | 117 | 0 | $0.0000 | 0.608 s |
| `gemini/gemini-flash-latest` *(excluded, smoke only)* | 10 | 0 | $0.0028 | 1.003 s |
| **TOTAL** | **10,656** | **29** | **$7.6221** | |

The TweetSumm re-run added **500 calls for $0.4208** (5 models x 100 dialogues,
**zero errors**), against a $3 budget for that step.

Budget: **$7.20**, which **exceeds the $6 ceiling** that was in force during the
main run. That was deliberate and user-directed: on 2026-08-01 the user added
Anthropic credit and instructed that Sonnet's few-shot arm be run, adding
$2.31. The `budget_cap_usd: 30` hard cap was never approached. Anthropic spend
totalled **$6.04**.

**Sonnet is the single largest line item ($4.42 of $7.20, 61%)** despite being
one of six models, because its post-4.7 tokenizer consumes ~34% more tokens
per call at 2x Haiku's per-token rate. Budget accordingly if this is rerun.

Caveat on `billed`: it is computed by `litellm.completion_cost`, whose bundled
price table is stale for three of our models (NOTES.md §9). Treat it as a close
estimate, not an invoice. Groq and Mistral figures are ~$0 because those ran on
free promotional tiers — **the paper's cost axis deliberately does not use this
column**; it uses published list prices (§4 below).

Errors: 29 of 10,156 (0.29%), all rate-limit exhaustion, all on models that
were retired early by design (28 Gemini, 1 Groq-8B). No error affected a
reported metric — failed calls are excluded from scoring and reported via
`error_rate`.

Cache: **10,091 files**, never deleted. Raw logs: **17,508 lines** across
`results/raw/{intent,summ}.jsonl` (more lines than calls because the logs are
append-only and consolidation re-appends; every consumer dedupes on call
identity).

---

## 2. Final lineup and what each model actually ran

| Model | Category | Zero-shot | Few-shot | Summ |
|---|---|---:|---:|---:|
| `claude-haiku-4-5-20251001` | proprietary | 1000 | 1000 | 100 |
| `claude-sonnet-5` | proprietary | 1000 | 1000 | 100 |
| `together_ai/…/Llama-3.3-70B-Instruct-Turbo` | open | 1000 | 1000 | 100 |
| `mistral/mistral-small-latest` | open | 1000 | 1000 | 100 |
| `groq/llama-3.1-8b-instant` | open | 1000 | — | 100 |
| `gemini/gemini-flash-lite-latest` | proprietary | 487\* | — | — |

\* supplementary subsample, excluded from the headline gap.

Four of six models ran the complete workload. `llama-3.1-8b` lacks few-shot for
a rate-limit reason (§4); Gemini is a capped subsample.

Category balance is satisfied: **2 proprietary (3 with Gemini) vs 3
open-weights**. Both conditions retained. Every reported metric rests on full
n=1000 (intent) or n=100 (summarization) except the clearly-labelled Gemini
supplementary rows.

---

## 3. Headline results

**Compliance gap — best proprietary vs best open-weights:**

| Task / condition | Best open | Best proprietary | Abs gap | Rel gap |
|---|---|---|---:|---:|
| Intent, zero-shot | llama-3.3-70B 0.746 | sonnet-5 **0.807** | 0.061 | **7.56%** |
| Intent, few-shot | llama-3.3-70B 0.757 | sonnet-5 **0.811** | 0.054 | **6.66%** |
| Intent, zero-shot (macro-F1) | llama-3.3-70B 0.7144 | sonnet-5 **0.798** | 0.0836 | **10.48%** |
| Intent, few-shot (macro-F1) | llama-3.3-70B 0.7253 | sonnet-5 **0.7931** | 0.0678 | **8.55%** |
| **Summarization — TweetSumm (primary)**, ROUGE-L | llama-3.3-70B **0.2198** | haiku-4.5 0.2060 | −0.0138 | **−6.70%** |
| Summarization — DialogSum (secondary), ROUGE-L | mistral **0.169** | sonnet-5 0.1615 | −0.0075 | **−4.64%** |

At published list prices the open side is ~2.7x cheaper on intent ($0.52 vs
$1.39 per 1,000 zero-shot queries; $0.65 vs $1.68 few-shot).

**Macro-F1 gaps run consistently wider than accuracy gaps** (10.5% vs 7.6%
zero-shot; 8.6% vs 6.7% few-shot). Banking77 has 77 classes, so macro-F1
weights rare intents equally with common ones: the open models lose more ground
on the tail than on the head. For a contact center that is arguably the more
decision-relevant number, since rare intents are the ones a misroute hurts
most. Worth reporting both rather than accuracy alone.

**Few-shot helps both sides but does not close the gap** — it lifts the 70B
from 0.746 to 0.757 and Sonnet from 0.807 to 0.811, so the gap narrows only
from 7.56% to 6.66%. Prompt engineering is not a substitute for model tier here.

**Summarization inverts the gap, and on the primary corpus it is statistically
real.** On TweetSumm — genuine customer-support dialogue — the best open-weights
model beats the best proprietary one by **+0.0138 ROUGE-L, 95% CI
[+0.0034, +0.0240], p≈0.009** (paired bootstrap, 10,000 resamples over the 100
shared dialogues). The interval excludes zero. On DialogSum the same comparison
gives **+0.0074, 95% CI [−0.0052, +0.0201], p≈0.245** — within noise. So
"summarization differences are within noise" is true out-of-domain and **false
in-domain**; do not state it as a blanket claim. Full detail in NOTES.md §11.

**The summarization ranking does not transfer across domains.**
`llama-3.3-70B` is 1st of 6 on TweetSumm (0.2198) and last on DialogSum
(0.1416); `mistral-small` is 1st on DialogSum and 4th on TweetSumm. Since
DialogSum's own top-2 difference is not significant, some of that reshuffle is
noise — but the 70B's movement far exceeds the DialogSum interval. Treat
ROUGE-based summarization rankings as corpus-specific.

**These gaps are an UPPER BOUND.** The 70B was served FP8-quantized, which can
only depress the open score, and `gap = proprietary − open`, so the measured
gap can overstate the true full-precision gap and never understate it. The
small-model skew on the open side pushes the same way; the two caveats compound.

Sanity check (CLAUDE.md step 10) passes: frontier proprietary zero-shot
accuracy is 0.777–0.807, inside the expected 60–90% band, with unparsed rates
of 0–0.6%. No near-zero score anywhere — label parsing is sound.

---

## 4. Substitutions and failures — everything that did not go to plan

**Models lost, with cause:**

1. `gpt-4o-mini` — OpenAI key authenticates but the account has **no credits**.
2. `claude-haiku` (original key) — the key in `~/Desktop/ACLsmith/.env.local`
   was **revoked** (`401`), not merely unfunded. Replaced by the user.
3. `gemini-2.0-flash` and all pinned 2.0/2.5 ids — this Google project has
   free-tier quota **provisioned at zero** (`limit: 0`). `gemini-2.5-flash*`
   additionally `404`s as "no longer available to new users".
4. `gemini-flash-latest` — resolves to `gemini-3.6-flash`, a **thinking model**
   that spends its whole completion budget reasoning and returns truncated
   fragments (0.0 accuracy, 100% unparsed on smoke test). Excluded rather than
   reported as a broken score.
5. `gemini-flash-lite-latest` — hit a hard **500 requests/day** free-tier cap at
   487 of 2,100 calls. Retained as a supplementary subsample.
6. `mistral-small` — no key initially; user supplied one, ran clean.
7. `groq/qwen3.6-27b`, `groq/gpt-oss-120b` — live but both thinking models,
   unusable under the task's `max_tokens=30`.
8. `together_ai/Qwen2.5-72B` — no key, no free tier.
9. **Cerebras — entirely unusable.** Added specifically to run llama-3.3-70b
   inside 24h. Two independent blockers: no Llama of any size in that account's
   catalogue (`404`), and `402 payment_required` on all three models it does
   have.
10. `meta-llama/Llama-3.3-70B-Instruct` (non-quantized, Together) — **not
    serverless** on this account, so the FP8 `-Turbo` build was used instead.

**Scope reductions, with cause:**

- **`n_test` 1500 → 1000, `n_dialogues` 120 → 100** — for free-tier daily
  request caps, not cost. Cache-safe: the test split is shuffled with
  `Random(42)` *before* slicing, so the 1,000 are a strict prefix of the 1,500
  and the few-shot exemplars are unchanged.
- **`claude-sonnet-5`: few-shot deferred, then completed.** Sonnet 5 uses the
  post-4.7 tokenizer, measured at **895 input tokens vs Haiku's 670 on the
  identical prompt (+34%)**. The two-condition workload priced at ~$4.27
  against ~$3.38 of credit then remaining and would have died mid-run, so the
  first pass ran zero-shot + summarization only ($2.11, inside the approved
  ~$2.50). The user added credit on 2026-08-01 and the few-shot arm then ran to
  completion: **1000/1000, 0 errors, $2.31** — within 3% of the $2.24 estimate.
  Sonnet consequently has the full workload and is gap-eligible in both
  conditions, and it took best-proprietary on few-shot (0.811) from Haiku
  (0.793), widening that gap from 4.54% to 6.66%.
- **`groq/llama-3.1-8b-instant`: no few-shot** — Groq meters it at **500,000
  tokens per DAY**. The 1,000 zero-shot calls (~500 tokens each) consumed
  essentially the whole daily budget (`Limit 500000, Used 499554`), leaving
  nothing for the ~645-token few-shot prompts; the remaining 967 projected to
  ~32 h of bucket refill. Zero-shot completed at full n and is gap-eligible.
- **`groq/llama-3.3-70b-versatile` moved to Together** — Groq's 1,000 req/day
  cap made it a ~52 h run. The user overrode the no-paid-endpoints rule for
  this model only. Cross-provider consistency check confirms the move did not
  change what was measured (17/17 label agreement; ROUGE-L delta 0.0018).

**Bugs found and fixed during the run:**

- Three of six models had **wrong prices in litellm 1.94.1**: Mistral Small
  ($0.06/$0.18 vs published $0.15/$0.60), Gemini Flash-Lite ($0.10/$0.40 vs
  $0.30/$2.50 — litellm assumes 2.5 Flash-Lite pricing), Together Llama-3.3-70B
  ($0.88 vs $1.04). All verified against vendor pages; the cost axis would have
  been materially wrong otherwise.
- **Summary JSONs were being clobbered** by concurrent per-model processes —
  `summ_outputs.json` held 2 of 7 models, which would have produced a silently
  incomplete expert rating sheet. Fixed with merge-on-write plus
  `src/rebuild_summaries.py`, which reconstructs all summaries from the raw
  logs (deterministic, no API calls, safe with partial models).
- **`claude-sonnet-5` had no price-table entry**, making its cost columns `nan`.
- **Preliminary banner** could stamp a report Final while runs were still in
  flight; now detects live runner processes directly.
- **llama-8b's stall was misdiagnosed, then corrected.** It was first attributed
  to the 6,000 tokens/min cap and a self-sustaining retry cascade, and "fixed"
  by raising the per-request delay. The provider's own error message — recovered
  from the raw log during the pre-publication secret sweep — shows the real
  cause was the **500,000 tokens/day** budget being already spent. No delay
  setting recovers an exhausted budget. The causal claim about raising the delay
  was not supported by evidence and has been withdrawn from NOTES.md §5; the
  surviving lesson is to size `n_calls x prompt_tokens` against the daily token
  cap before starting, and to order conditions by cost when they share a budget.

**Nothing was fabricated, interpolated, or back-filled.** Every number in
`tables.md` traces to a row in `results/raw/*.jsonl`.

---

## 5. Cost axis — read this before using the charts

The scatter charts plot **hosted API list price**, not what this study was
billed, because four models ran on free promotional tiers and billing them at
$0 would make the money chart an artifact of promotions. Prices, sources and
retrieval date are in NOTES.md §9 and repeated in `tables.md`.

Token counts are **estimated** — the providers' `usage` fields were not
persisted, so prompts were reconstructed exactly as the runners built them and
tokenized with `litellm.token_counter`. Deterministic and applied identically
to every model, so sound for comparison; quote absolute values as estimates.

This is **not** self-hosting TCO. GPU capital/rental, engineering time and idle
capacity are the real costs a privacy-constrained enterprise trades against,
and none are on this axis. That belongs in Limitations.

**Known expiry:** Sonnet 5's $2/$10 is *introductory pricing through
2026-08-31*; from 2026-09-01 it lists at $3/$15, which would raise its
cost-per-1k by 50%. Re-derive the axis if publishing after that date.

---

## 6. Acceptance criteria

| Artifact | Exists | Notes |
|---|---|---|
| `results/tables.md` | **YES** | per-model per-condition metrics + compliance-gap table + supplementary section + cost basis |
| `results/charts/accuracy_vs_cost.png` | **YES** | the money chart, category-coloured, list-price axis |
| `results/charts/rougeL_vs_cost.png` | **YES** | category-coloured |
| `results/charts/zs_vs_fs.png` | **YES** | category-coloured tick labels |
| `results/expert_rating_sheet.csv` | **YES** | 40 rows, 5 models x 8, blinded, rating columns blank |
| `results/expert_rating_key.csv` | **YES** | **blinding key — withhold from the rater** |
| `results/NOTES.md` | **YES** | 10 sections; Methods-ready |
| `results/RUN_REPORT.md` | **YES** | this file |
| `cache/` | **9,091 files** | never deleted |
| `results/raw/` | **16,508 lines** | full call-level record |

Cost and median latency per 1,000 calls per model: in `tables.md`
(`list_cost_per_1k`, `billed_cost_per_1k`, `median_latency_s`).

**Rating sheet integrity:** verified 8 summaries per model across 5 models, all
four rating columns blank, and no model identity anywhere in the sheet (the one
regex hit was the word "altogether" in a customer dialogue). Two models were
deliberately excluded and the reason is printed by the script: the Groq 70B
baseline (same weights as the Together entry — including both would show the
rater the same model twice under two codes) and Gemini (3 summaries, cannot
fill a balanced quota).

---

## 7. For the Methods section — cite these, not the aliases

| Model | Serving provider | Concrete snapshot | Precision |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | Anthropic | `claude-haiku-4-5-20251001` | vendor default |
| `claude-sonnet-5` | Anthropic | `claude-sonnet-5` | vendor default |
| `gemini/gemini-flash-lite-latest` | Google AI Studio | **`gemini-3.5-flash-lite`** | vendor default |
| `together_ai/…Llama-3.3-70B-Instruct-Turbo` | Together AI | `Llama-3.3-70B-Instruct-Turbo` | **FP8** |
| `mistral/mistral-small-latest` | Mistral | Mistral Small 4 | vendor default |
| `groq/llama-3.1-8b-instant` | Groq | `llama-3.1-8b-instant` | Groq-quantized |

The Gemini alias does not report its own version through litellm or the raw
logs — `gemini-3.5-flash-lite` comes from the `modelVersion` field in the raw
API response. Aliases can hot-swap under a rerun.

**Datasets.** Banking77 via `legacy-datasets/banking77` (test 3,080 → seed-42
shuffle → first 1,000).

Summarization uses two corpora:

- **PRIMARY — TweetSumm** (<https://github.com/guyfe/Tweetsumm>), test split:
  110 dialogues referencing 1,152 tweet IDs, **all 1,152 resolved, 0 dialogues
  dropped**; seed-42 shuffle → first 100. Reconstructed locally from tweet IDs
  plus Kaggle's `twcs.csv` using the upstream `TweetSumProcessor`. Reference =
  the **first** of three human abstractive annotations, sentences joined.
  Genuine customer-support dialogue, so this is the in-domain measurement.
  **Licensing:** dataset is CDLA-Sharing-1.0 with text sourced from Kaggle, so
  no dialogue text, `twcs.csv`, or derived summarization records are in the
  public repo — only the loader and IDs.
- **SECONDARY — DialogSum** (`knkarthick/dialogsum`), test split, seed-42
  shuffle → first 100. General-domain daily conversation, retained as a
  cross-domain robustness check on whether the model ordering is
  corpus-specific. Originally the primary corpus, when TweetSumm was
  unavailable (gated *and* script-based via DialogStudio).

Determinism: `temperature=0`, `seed=42` where the provider accepts it.

---

## 8. Known limitations to carry into the paper

1. **Proprietary side is one vendor, two tiers** (Haiku/Sonnet). The gap is
   really "open weights vs *Anthropic*". Mitigated by the Gemini subsample
   (0.797 at n=487, landing between the two Anthropic models — genuine
   second-vendor corroboration) and by tier-scaling symmetry (8B→24B→70B
   against Haiku→Sonnet).
2. **The 70B ran FP8-quantized**, so the reported gap is an upper bound.
3. **Summarization is still ROUGE-based, and ROUGE is weak here.** The primary
   corpus is now TweetSumm (real customer support), which removes the earlier
   domain-mismatch objection — but ROUGE still rewards surface overlap rather
   than whether a summary is *usable at agent handoff*, and each dialogue has
   three human references while only the first is scored against. Multi-
   reference ROUGE would be strictly better and is available in the raw
   annotations; it is not implemented here. The blinded expert rating sheet,
   now built from TweetSumm, remains the instrument for the usability claim.
4. **`llama-3.1-8b` lacks few-shot** (Groq token-rate cap), so it contributes
   to zero-shot only. Both proprietary models and both larger open models ran
   the full workload, so the headline gap is unaffected.
5. **Cost axis is list price, not TCO** (§5).
6. **Aliases can hot-swap**; `mistral-small-latest` and
   `gemini-flash-lite-latest` are moving targets.
