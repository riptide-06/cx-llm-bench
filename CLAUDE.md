# CX-LLM-Bench: The Compliance Gap in Contact Center AI

## Context
Research project producing a paper for Computers (MDPI):
"Benchmarking open-weights vs. proprietary LLMs for privacy-constrained
contact center automation." Research question: how much task performance do
privacy-constrained enterprises (esp. healthcare) sacrifice by restricting
contact center AI to self-hostable open-weights models instead of
proprietary API models? Tasks: intent classification (Banking77) and
conversation summarization (TweetSumm, fallback dialogsum). The human
(Tarun) is busy with classwork; get the pipeline running with minimal
check-ins. The open-vs-proprietary comparison IS the paper, so the model
lineup in config.yaml must stay balanced (>=2 models per category).

## Your mission (in order)

### Phase 0: Setup and smoke test (~15 min)
1. `pip install -r requirements.txt` (use a venv)
2. Check which API keys exist (OPENAI_API_KEY, ANTHROPIC_API_KEY,
   GEMINI_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY).
   Comment out models whose keys are missing, keeping >=2 proprietary and
   >=2 open-weights. Verify each model ID is live; swap deprecated IDs for
   the closest same-family equivalent and record swaps in results/NOTES.md.
   Keep open_weights_models in sync with any changes.
3. Smoke test: `python src/run_intent.py --limit 20` and
   `python src/run_summ.py --limit 5`. Fix dataset-loading/parsing errors.
   TweetSumm notes are in run_summ.py; fallback to knkarthick/dialogsum is
   acceptable, note it in NOTES.md.
4. STOP. Report smoke results, fixes, active models per category, and
   estimated full-run cost. Wait for go-ahead; hard stop if estimate > $30.

### Phase 1: Full runs (unattended; cache makes reruns free)
5. `python src/run_intent.py` (1500 queries, zero-shot AND few-shot k=5)
6. `python src/run_summ.py` (120 dialogues)
7. On repeated rate-limit errors: slow down that model and rerun; the cache
   skips completed calls.

### Phase 2: Analysis and outputs
8. Modify src/report.py before running it:
   a. Color-code the accuracy-vs-cost scatter by category (open-weights vs
      proprietary, using open_weights_models from config).
   b. Add a "compliance gap" table: best open vs best proprietary per
      task/condition, absolute and relative deltas (accuracy, macro-F1,
      ROUGE-L), plus cost per 1,000 calls for each.
9. `python src/report.py` -> results/tables.md, results/charts/*.png
10. Sanity-check: frontier proprietary models should land roughly 60-90%
    accuracy on Banking77 zero-shot. A near-0 score means broken output
    parsing — fix the label matching, don't report it.
11. results/NOTES.md: anomalies, dataset versions/splits, exact model
    snapshot IDs, any ID swaps, total cost, total calls.
12. Expert rating sheet: randomly sample 40 summaries (balanced across
    models, seed=42) into results/expert_rating_sheet.csv with columns:
    sample_id, dialogue, model (BLINDED — model key in a separate file),
    summary, usefulness_1to5, missing_critical_info_yn,
    would_trust_in_handoff_yn, comments.

## Hard rules
- NEVER fabricate, interpolate, or "fix" results. Failures go in NOTES.md.
- Cache every call; never delete cache/. Keep raw logs in results/raw/.
- Budget cap $30 without explicit approval.
- temperature=0, fixed seeds. Determinism wherever possible.

## Acceptance criteria
- tables.md: per-model per-condition metrics + the compliance-gap table
- Cost and median latency per 1,000 calls per model
- Charts incl. category-colored accuracy-vs-cost scatter (the money chart)
- expert_rating_sheet.csv ready for the domain expert
- NOTES.md complete enough to write the Methods section from
