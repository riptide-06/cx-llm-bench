# Results

_Final: every configured model completed both tasks._

## Intent classification (Banking77)

Full-sample results (n = 1000 per condition). These are the only rows used for the compliance gap.

| model                                               | category     | condition   |    n |   n_scored |   accuracy |   macro_f1 |   unparsed_rate |   error_rate |   n_calls |   billed_cost_per_1k |   median_latency_s |   priced_frac |   mean_input_tokens |   mean_output_tokens |   n_priced_calls |   list_cost_per_1k |
|:----------------------------------------------------|:-------------|:------------|-----:|-----------:|-----------:|-----------:|----------------:|-------------:|----------:|---------------------:|-------------------:|--------------:|--------------------:|---------------------:|-----------------:|-------------------:|
| mistral/mistral-small-latest                        | open-weights | few_shot    | 1000 |       1000 |      0.716 |     0.694  |           0.002 |            0 |      1000 |               0.0042 |              0.527 |             1 |               622.7 |                  3.9 |             1000 |             0.0957 |
| together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo | open-weights | few_shot    | 1000 |       1000 |      0.757 |     0.7253 |           0.002 |            0 |      1000 |               0.5962 |              0.637 |             1 |               622.7 |                  3.7 |             1000 |             0.6515 |
| claude-haiku-4-5-20251001                           | proprietary  | few_shot    | 1000 |       1000 |      0.793 |     0.7686 |           0.006 |            0 |      1000 |               0.846  |              0.664 |             1 |               809.9 |                  5.8 |             1000 |             0.8388 |
| claude-sonnet-5                                     | proprietary  | few_shot    | 1000 |       1000 |      0.811 |     0.7931 |           0.001 |            0 |      1000 |               2.3106 |              1.22  |             1 |               809.9 |                  5.7 |             1000 |             1.6772 |
| groq/llama-3.1-8b-instant                           | open-weights | zero_shot   | 1000 |       1000 |      0.602 |     0.5833 |           0.004 |            0 |      1000 |               0      |              0.283 |             0 |               500.7 |                  3.7 |             1000 |             0.0253 |
| mistral/mistral-small-latest                        | open-weights | zero_shot   | 1000 |       1000 |      0.699 |     0.6805 |           0.003 |            0 |      1000 |               0.0044 |              0.542 |             1 |               500.7 |                  3.9 |             1000 |             0.0774 |
| together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo | open-weights | zero_shot   | 1000 |       1000 |      0.746 |     0.7144 |           0     |            0 |      1000 |               0.4864 |              0.512 |             1 |               500.7 |                  3.8 |             1000 |             0.5246 |
| claude-haiku-4-5-20251001                           | proprietary  | zero_shot   | 1000 |       1000 |      0.777 |     0.7572 |           0.001 |            0 |      1000 |               0.7117 |              0.688 |             1 |               668.9 |                  5.7 |             1000 |             0.6974 |
| claude-sonnet-5                                     | proprietary  | zero_shot   | 1000 |       1000 |      0.807 |     0.798  |           0.029 |            0 |      1000 |               1.8991 |              1.258 |             1 |               668.9 |                  5.6 |             1000 |             1.394  |


### Supplementary: reduced-sample models (EXCLUDED from the compliance gap)

Scored on a smaller subsample than the models above, so not directly comparable to them. The test set is shuffled with `Random(42)` *before* slicing, so a truncated run is a valid random subsample rather than a biased prefix — but its confidence interval is wider and it must be reported at its own n. Reasons:

- `gemini/gemini-flash-lite-latest` — zero-shot intent only, at a reduced n (~459 of 1000); Gemini's free tier caps gemini-3.5-flash-lite at 500 requests/day, so the full workload needs 4+ days. The completed calls are a valid seed-42 random subsample.
- `groq/llama-3.1-8b-instant` — zero-shot intent (full n=1000) plus full summarization; few-shot omitted. Groq's free tier meters this model at 500,000 tokens per DAY, and the 1000 zero-shot calls (~500 tokens each) consumed essentially all of it ("Limit 500000, Used 499554"), leaving nothing for the ~645-token few-shot prompts; the remaining 967 projected to ~32h of refill. Zero-shot completed at full n and is gap-eligible. See NOTES.md section 5.

| model                           | category     | condition   |   n |   n_scored |   accuracy |   macro_f1 |   unparsed_rate |   error_rate |   n_calls |   billed_cost_per_1k |   median_latency_s |   priced_frac |   mean_input_tokens |   mean_output_tokens |   n_priced_calls |   list_cost_per_1k |
|:--------------------------------|:-------------|:------------|----:|-----------:|-----------:|-----------:|----------------:|-------------:|----------:|---------------------:|-------------------:|--------------:|--------------------:|---------------------:|-----------------:|-------------------:|
| groq/llama-3.1-8b-instant       | open-weights | few_shot    |  34 |         33 |     0.4545 |     0.4542 |          0.0606 |       0.0294 |        33 |               0      |              0.334 |             0 |               621.2 |                 17.5 |               33 |             0.0325 |
| gemini/gemini-flash-latest      | proprietary  | few_shot    |   5 |          5 |     0      |     0      |          1      |       0      |         5 |               0.3004 |              0.993 |             1 |               621.4 |                  1.4 |                5 |           nan      |
| gemini/gemini-flash-lite-latest | proprietary  | few_shot    |   5 |          5 |     1      |     1      |          0      |       0      |         5 |               0.0814 |              0.451 |             1 |               621.4 |                  5   |                5 |             0.1989 |
| groq/llama-3.3-70b-versatile    | open-weights | zero_shot   |  17 |         17 |     0.8235 |     0.8    |          0      |       0      |        17 |               0      |              0.384 |             0 |               498.2 |                  3.9 |               17 |             0.297  |
| gemini/gemini-flash-latest      | proprietary  | zero_shot   |   5 |          5 |     0      |     0      |          1      |       0      |         5 |               0.2615 |              1.003 |             1 |               499.4 |                  0.6 |                5 |           nan      |
| gemini/gemini-flash-lite-latest | proprietary  | zero_shot   | 487 |        459 |     0.7974 |     0.7749 |          0      |       0.0575 |       459 |               0.0675 |              0.482 |             1 |               500.3 |                  3.8 |              459 |             0.1596 |


## Summarization — TweetSumm (PRIMARY, in-domain customer support)

| model                                               | category     |   rouge1 |   rouge2 |   rougeL |   n_scored |   error_rate |   n_calls |   billed_cost_per_1k |   median_latency_s |   priced_frac |   mean_input_tokens |   mean_output_tokens |   n_priced_calls |   list_cost_per_1k |
|:----------------------------------------------------|:-------------|---------:|---------:|---------:|-----------:|-------------:|----------:|---------------------:|-------------------:|--------------:|--------------------:|---------------------:|-----------------:|-------------------:|
| groq/llama-3.1-8b-instant                           | open-weights |   0.3007 |   0.0834 |   0.2069 |        100 |            0 |       100 |               0      |              0.548 |             0 |               376.9 |                 99.8 |              100 |             0.0268 |
| mistral/mistral-small-latest                        | open-weights |   0.2999 |   0.0691 |   0.2053 |        100 |            0 |       100 |               0.0407 |              1.258 |             1 |               377   |                 93.8 |              100 |             0.1128 |
| together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo | open-weights |   0.3086 |   0.0874 |   0.2198 |        100 |            0 |       100 |               0.4592 |              1.256 |             1 |               376.9 |                 97.3 |              100 |             0.4932 |
| claude-haiku-4-5-20251001                           | proprietary  |   0.2994 |   0.0716 |   0.206  |        100 |            0 |       100 |               0.937  |              1.98  |             1 |               394.4 |                 93.5 |              100 |             0.8618 |
| claude-sonnet-5                                     | proprietary  |   0.2945 |   0.0661 |   0.2028 |        100 |            0 |       100 |               2.7712 |              3.421 |             1 |               394.4 |                101.5 |              100 |             1.8039 |


## Summarization — dialogsum (SECONDARY, cross-domain robustness check)

_DialogSum is general-domain daily conversation, not contact center data. It is reported as a robustness check on whether the TweetSumm ordering holds out of domain — not as the headline summarization result._

| model                                               | category     |   rouge1 |   rouge2 |   rougeL |   n_scored |   error_rate |   n_calls |   billed_cost_per_1k |   median_latency_s |   priced_frac |   mean_input_tokens |   mean_output_tokens |   n_priced_calls |   list_cost_per_1k |
|:----------------------------------------------------|:-------------|---------:|---------:|---------:|-----------:|-------------:|----------:|---------------------:|-------------------:|--------------:|--------------------:|---------------------:|-----------------:|-------------------:|
| groq/llama-3.1-8b-instant                           | open-weights |   0.1993 |   0.057  |   0.1423 |        100 |            0 |       100 |               0      |              0.467 |             0 |               261.9 |                 90   |              100 |             0.0203 |
| groq/llama-3.3-70b-versatile                        | open-weights |   0.1949 |   0.0504 |   0.1434 |        100 |            0 |       100 |               0      |              0.634 |             0 |               261.9 |                 73.7 |              100 |             0.2128 |
| mistral/mistral-small-latest                        | open-weights |   0.2327 |   0.0633 |   0.169  |        100 |            0 |       100 |               0.0291 |              1.107 |             1 |               261.9 |                 75.5 |              100 |             0.0846 |
| together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo | open-weights |   0.1934 |   0.0489 |   0.1416 |        100 |            0 |       100 |               0.3332 |              2.7   |             1 |               261.9 |                 72.3 |              100 |             0.3476 |
| claude-haiku-4-5-20251001                           | proprietary  |   0.1953 |   0.0482 |   0.1459 |        100 |            0 |       100 |               0.6564 |              1.495 |             1 |               277.7 |                 65.4 |              100 |             0.6046 |
| claude-sonnet-5                                     | proprietary  |   0.2257 |   0.0631 |   0.1615 |        100 |            0 |       100 |               2.0626 |              2.321 |             1 |               277.7 |                 81.6 |              100 |             1.3712 |


### Supplementary: reduced-sample models (EXCLUDED from the compliance gap)

| model                           | category    |   rouge1 |   rouge2 |   rougeL |   n_scored |   error_rate |   n_calls |   billed_cost_per_1k |   median_latency_s |   priced_frac |   mean_input_tokens |   mean_output_tokens |   n_priced_calls |   list_cost_per_1k |
|:--------------------------------|:------------|---------:|---------:|---------:|-----------:|-------------:|----------:|---------------------:|-------------------:|--------------:|--------------------:|---------------------:|-----------------:|-------------------:|
| gemini/gemini-flash-lite-latest | proprietary |   0.1578 |   0.0185 |   0.0891 |          3 |            0 |         3 |               0.0932 |              0.638 |             1 |                 662 |                   54 |                3 |             0.3336 |


## Compliance gap: best proprietary vs. best open-weights

`abs_gap` = proprietary - open (positive favors proprietary). `rel_gap_pct` = abs_gap / proprietary score, in percent. Cost columns are hosted API list price (see below).

> **These gaps are an UPPER BOUND on the true full-precision gap.** The 70B was served FP8-quantized (Together's non-quantized build is not serverless on this account), and quantization can only depress the open-weights score. Since `gap = proprietary - open`, a depressed `open` inflates the gap: `gap_measured >= gap_true`. FP8 can overstate the gap, never understate it. The small-model skew on the open side (NOTES.md §3) pushes the same direction, so the two caveats compound. Quote these figures with the upper-bound qualifier attached.

| task                                 | condition   | metric   | best_open                                           |   open_score |   open_list_cost_per_1k | best_proprietary          |   prop_score |   prop_list_cost_per_1k |   abs_gap |   rel_gap_pct |
|:-------------------------------------|:------------|:---------|:----------------------------------------------------|-------------:|------------------------:|:--------------------------|-------------:|------------------------:|----------:|--------------:|
| intent (Banking77)                   | few_shot    | accuracy | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.757  |                  0.6515 | claude-sonnet-5           |       0.811  |                  1.6772 |    0.054  |          6.66 |
| intent (Banking77)                   | few_shot    | macro_f1 | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.7253 |                  0.6515 | claude-sonnet-5           |       0.7931 |                  1.6772 |    0.0678 |          8.55 |
| intent (Banking77)                   | zero_shot   | accuracy | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.746  |                  0.5246 | claude-sonnet-5           |       0.807  |                  1.394  |    0.061  |          7.56 |
| intent (Banking77)                   | zero_shot   | macro_f1 | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.7144 |                  0.5246 | claude-sonnet-5           |       0.798  |                  1.394  |    0.0836 |         10.48 |
| summarization (TweetSumm)            | -           | rougeL   | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.2198 |                  0.4932 | claude-haiku-4-5-20251001 |       0.206  |                  0.8618 |   -0.0138 |         -6.7  |
| summarization (TweetSumm)            | -           | rouge1   | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.3086 |                  0.4932 | claude-haiku-4-5-20251001 |       0.2994 |                  0.8618 |   -0.0092 |         -3.07 |
| summarization (TweetSumm)            | -           | rouge2   | together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo |       0.0874 |                  0.4932 | claude-haiku-4-5-20251001 |       0.0716 |                  0.8618 |   -0.0158 |        -22.07 |
| summarization (dialogsum, secondary) | -           | rougeL   | mistral/mistral-small-latest                        |       0.169  |                  0.0846 | claude-sonnet-5           |       0.1615 |                  1.3712 |   -0.0075 |         -4.64 |
| summarization (dialogsum, secondary) | -           | rouge1   | mistral/mistral-small-latest                        |       0.2327 |                  0.0846 | claude-sonnet-5           |       0.2257 |                  1.3712 |   -0.007  |         -3.1  |
| summarization (dialogsum, secondary) | -           | rouge2   | mistral/mistral-small-latest                        |       0.0633 |                  0.0846 | claude-sonnet-5           |       0.0631 |                  1.3712 |   -0.0002 |         -0.32 |


## Cost basis

`list_cost_per_1k` is **hosted API list price**, not what this study was billed. Four of the five models ran on free promotional tiers, so `billed_cost_per_1k` is ~$0 for them and would make the cost axis an artifact of promotions rather than an economic comparison. It is also **not** self-hosting TCO (GPU capital/rental, engineering time, idle capacity) — that belongs in the paper's limitations.

Token counts are **estimated**: the providers' `usage` fields were not persisted, so prompts were reconstructed exactly as the runners built them and tokenized with `litellm.token_counter`, along with the stored output text. For non-OpenAI models this is an approximation (a few percent), excludes chat-template scaffolding, and is applied identically to every model — sound for comparison, quote absolute values as estimates. See `src/list_price.py` and NOTES.md §9.

| Model | Priced as | Input $/M | Output $/M | Source |
|---|---|---:|---:|---|
| `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` | 1.00 | 5.00 | https://platform.claude.com/docs/en/about-claude/pricing (Claude Haiku 4.5) |
| `claude-sonnet-5` | `claude-sonnet-5` | 2.00 | 10.00 | https://platform.claude.com/docs/en/about-claude/pricing (Claude Sonnet 5, introductory rate through 2026-08-31) |
| `gemini/gemini-flash-lite-latest` | `gemini-3.5-flash-lite` | 0.30 | 2.50 | https://ai.google.dev/gemini-api/docs/pricing (Gemini 3.5 Flash-Lite) |
| `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `meta-llama/Llama-3.3-70B-Instruct-Turbo (FP8)` | 1.04 | 1.04 | https://www.together.ai/pricing (Llama 3.3 70B) |
| `groq/llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` | 0.59 | 0.79 | https://groq.com/pricing (Llama 3.3 70B Versatile 128k) |
| `groq/llama-3.1-8b-instant` | `llama-3.1-8b-instant` | 0.05 | 0.08 | https://groq.com/pricing (Llama 3.1 8B Instant 128k) |
| `mistral/mistral-small-latest` | `mistral-small-latest (Mistral Small 4)` | 0.15 | 0.60 | https://mistral.ai/pricing/api (Mistral Small 4) |

All prices retrieved **2026-07-31**.
