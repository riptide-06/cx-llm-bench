"""Aggregate results into markdown tables + charts for the paper."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CHARTS = RES / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

OPEN_COLOR = "#2077B4"
PROP_COLOR = "#D62728"


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def category_of(model: str, open_models: set[str]) -> str:
    return "open-weights" if model in open_models else "proprietary"


# Full litellm ids are far too long to annotate a scatter point with — the
# Together id alone overran the axis. Short labels keep the serving provider
# and the quantization, which are the parts a reader needs.
SHORT_NAMES = {
    "claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "gemini/gemini-flash-lite-latest": "gemini-3.5-flash-lite",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama-3.3-70B (Together, FP8)",
    "groq/llama-3.3-70b-versatile": "llama-3.3-70B (Groq)",
    "groq/llama-3.1-8b-instant": "llama-3.1-8B (Groq)",
    "mistral/mistral-small-latest": "mistral-small",
}


def short_name(model: str) -> str:
    return SHORT_NAMES.get(model, model.split("/")[-1])


def cost_latency_from_raw(name, key_fields):
    """Per (model, condition) cost and latency from the raw log.

    The log is append-only and reruns re-append every row, so dedupe on the
    call identity first — otherwise a rerun double-counts cost. Latency is
    taken from cached rows too, since a cached row carries the latency that
    was measured when the call was originally made; ignoring them would leave
    a fully-cached rerun with no latency data at all.
    """
    p = RES / "raw" / f"{name}.jsonl"
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
        ident = tuple(r.get(k) for k in key_fields)
        seen[ident] = r  # last write wins; identical calls are deduped

    stats = defaultdict(lambda: {"cost": 0.0, "lat": [], "n": 0, "priced": 0})
    for r in seen.values():
        if r.get("error"):
            continue  # failed calls have no cost/latency to report
        s = stats[(r["model"], r.get("condition", "all"))]
        s["n"] += 1
        # `is not None` matters: a free-tier call legitimately costs 0.0, which
        # must not be counted as "litellm could not price this call"
        if r.get("cost_usd") is not None:
            s["cost"] += r["cost_usd"]
            s["priced"] += 1
        if r.get("latency_s") is not None:
            s["lat"].append(r["latency_s"])

    out = {}
    for (m, cond), s in stats.items():
        lat = sorted(s["lat"])
        out[(m, cond)] = {
            "n_calls": s["n"],
            # What we were ACTUALLY billed. Mostly $0.00 here because four of
            # the five models ran on free promotional tiers — which is why the
            # charts use list_cost_per_1k instead. Kept for transparency.
            "billed_cost_per_1k": round(s["cost"] / s["n"] * 1000, 4) if s["n"] else None,
            "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
            # flags a model whose pricing litellm could not resolve, so a $0
            # cost is not mistaken for "free"
            "priced_frac": round(s["priced"] / s["n"], 3) if s["n"] else None,
        }
    return out


def ensure_cols(df, cols):
    """Guarantee `cols` exist so downstream dropna/plotting cannot KeyError.

    When every call for a model failed there are no priced rows at all, so the
    cost/latency merge contributes no columns.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _best(df, metric, category, higher_is_better=True):
    sub = df[(df.category == category) & df[metric].notna()]
    if not len(sub):
        return None
    idx = sub[metric].idxmax() if higher_is_better else sub[metric].idxmin()
    return sub.loc[idx]


def compliance_gap_rows(df, metrics, task_label, cond_col=None):
    """best-proprietary vs best-open per metric, absolute and relative.

    Only FULL-n rows are eligible. A model scored on a smaller subsample (e.g.
    Gemini, stopped by its free-tier daily cap) is not comparable to one scored
    on the whole set, so including it could hand the headline gap to a model
    measured on different data. Those rows are reported separately instead.
    """
    if "full_n" in df.columns:
        df = df[df.full_n]
    rows = []
    groups = df.groupby(cond_col) if cond_col else [("", df)]
    for cond, g in groups:
        for metric in metrics:
            if metric not in g.columns:
                continue
            bo = _best(g, metric, "open-weights")
            bp = _best(g, metric, "proprietary")
            if bo is None or bp is None:
                continue
            ov, pv = float(bo[metric]), float(bp[metric])
            rows.append({
                "task": task_label,
                "condition": cond or "-",
                "metric": metric,
                "best_open": bo.model,
                "open_score": round(ov, 4),
                "open_list_cost_per_1k": bo.get("list_cost_per_1k"),
                "best_proprietary": bp.model,
                "prop_score": round(pv, 4),
                "prop_list_cost_per_1k": bp.get("list_cost_per_1k"),
                "abs_gap": round(pv - ov, 4),
                "rel_gap_pct": round((pv - ov) / pv * 100, 2) if pv else None,
            })
    return rows


def main():
    cfg = load_config()
    open_models = set(cfg.get("open_weights_models") or [])

    # Cost axis = published hosted-API list price, NOT what we were billed.
    # Four of five models ran on free promotional tiers, so billed cost would
    # plot them at $0 and make the chart an artifact of promo pricing.
    import list_price
    lp = list_price.list_cost_per_1k()

    # A model is "complete" only when it has every expected call for both
    # tasks. Anything short of that makes the whole report preliminary, since
    # a partially-run model's accuracy is measured on a different sample than
    # its peers and the compliance gap would be comparing unequal subsets.
    want_i = cfg["intent"]["n_test"] * 2
    want_s = cfg["summ"]["n_dialogues"]
    done = defaultdict(lambda: {"intent": 0, "summ": 0})
    for task, keys in (("intent", ("model", "condition", "idx")),
                       ("summ", ("model", "id"))):
        p = RES / "raw" / f"{task}.jsonl"
        if not p.exists():
            continue
        seen = set()
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            seen.add(tuple(r.get(k) for k in keys))
        for ident in seen:
            done[ident[0]][task] += 1
    # A deliberately-reduced model is not "missing" — nothing more is coming.
    partial = cfg.get("partial_models") or {}
    missing = [f"{m} ({done[m]['intent']}/{want_i} intent, "
               f"{done[m]['summ']}/{want_s} summ)"
               for m in cfg["models"]
               if m not in partial
               and (done[m]["intent"] < want_i or done[m]["summ"] < want_s)]

    # `partial_models` says "expect less than the full workload", but it cannot
    # say "and that model has finished". A model still mid-run would otherwise
    # be exempted and the report stamped Final while data is still arriving —
    # so check for live runners directly.
    try:
        import subprocess
        # Match the interpreter+script, not the bare script name: a watcher
        # shell whose own command line contains "src/run_summ.py" would
        # otherwise count as a live runner and stamp the FINAL report
        # preliminary forever.
        live = subprocess.run(
            ["pgrep", "-f", "Python src/run_(intent|summ)[.]py"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        live = ""
    if live:
        missing.append(
            f"{len(live.splitlines())} runner process(es) still executing — "
            "data is still arriving")

    lines = ["# Results\n"]
    if missing:
        lines.append(
            "> **PRELIMINARY — DO NOT PUBLISH THESE NUMBERS.**\n"
            ">\n"
            "> One or more configured models have not finished running, so the\n"
            "> lineup is incomplete and the compliance gap below is **not\n"
            "> final**. Incomplete models:\n>\n"
            + "".join(f"> - `{m}`\n" for m in missing)
            + ">\n> Regenerate once every model has completed.\n")
    else:
        lines.append("_Final: every configured model completed both tasks._\n")
    intent = json.loads((RES / "intent_summary.json").read_text()) \
        if (RES / "intent_summary.json").exists() else {}

    def _load(name):
        p = RES / name
        return json.loads(p.read_text()) if p.exists() else {}

    # PRIMARY first — order matters, it drives the order of both the tables and
    # the compliance-gap rows.
    SUMM_DATASETS = [
        {"data": _load("summ_tweetsumm_summary.json"), "key": "tweetsumm",
         "raw": "summ_tweetsumm", "chart": "rougeL_vs_cost.png",
         "primary": True},
        {"data": _load("summ_summary.json"), "key": "all",
         "raw": "summ", "chart": "rougeL_vs_cost_dialogsum.png",
         "primary": False},
    ]

    icl = cost_latency_from_raw("intent", ("model", "condition", "idx"))
    scl = {}
    for d in SUMM_DATASETS:
        for (m, _c), v in cost_latency_from_raw(d["raw"], ("model", "id")).items():
            scl[(m, d["key"])] = v

    gap_rows = []

    if intent:
        rows = []
        for m, conds in intent.items():
            for cond, s in conds.items():
                rows.append({"model": m,
                             "category": category_of(m, open_models),
                             "condition": cond, **s,
                             **icl.get((m, cond), {}),
                             **lp.get((m, cond), {})})
        df = pd.DataFrame(rows).sort_values(["condition", "category", "model"])
        df = ensure_cols(df, ["accuracy", "macro_f1", "list_cost_per_1k",
                              "billed_cost_per_1k", "median_latency_s",
                              "n_calls", "priced_frac", "mean_input_tokens",
                              "mean_output_tokens"])
        df["full_n"] = df["n"] >= cfg["intent"]["n_test"]
        lines.append("## Intent classification (Banking77)\n")
        full = df[df.full_n]
        sub = df[~df.full_n]
        lines.append(f"Full-sample results (n = {cfg['intent']['n_test']} per "
                     "condition). These are the only rows used for the "
                     "compliance gap.\n")
        lines.append(full.drop(columns=["full_n"]).to_markdown(index=False) + "\n")
        if len(sub):
            lines.append("\n### Supplementary: reduced-sample models "
                         "(EXCLUDED from the compliance gap)\n")
            lines.append(
                "Scored on a smaller subsample than the models above, so not "
                "directly comparable to them. The test set is shuffled with "
                "`Random(42)` *before* slicing, so a truncated run is a valid "
                "random subsample rather than a biased prefix — but its "
                "confidence interval is wider and it must be reported at its "
                "own n. Reasons:\n")
            for m, why in (cfg.get("partial_models") or {}).items():
                if m in set(sub.model):
                    lines.append(f"- `{m}` — {' '.join(str(why).split())}")
            lines.append("")
            lines.append(sub.drop(columns=["full_n"]).to_markdown(index=False) + "\n")
        gap_rows += compliance_gap_rows(
            df, ["accuracy", "macro_f1"], "intent (Banking77)", "condition")

        # money chart: accuracy vs cost, colored by category
        # Full-n only, matching the compliance gap: plotting a model scored on
        # a different-sized sample alongside the others would invite a
        # like-for-like reading of points that are not like-for-like.
        zs = df[(df.condition == "zero_shot") & df.full_n].dropna(
            subset=["list_cost_per_1k", "accuracy"])
        if len(zs):
            plt.figure(figsize=(7.5, 5.5))
            for cat, color in [("open-weights", OPEN_COLOR),
                               ("proprietary", PROP_COLOR)]:
                c = zs[zs.category == cat]
                if len(c):
                    plt.scatter(c.list_cost_per_1k, c.accuracy, s=90, c=color,
                                label=cat, edgecolors="black", linewidths=0.5,
                                zorder=3)
            for _, r in zs.iterrows():
                plt.annotate(short_name(r.model),
                             (r.list_cost_per_1k, r.accuracy),
                             fontsize=8, xytext=(6, 6),
                             textcoords="offset points")
            plt.xlabel("Cost per 1,000 queries (USD, hosted API list price)")
            plt.ylabel("Accuracy (zero-shot)")
            plt.title("Intent classification: accuracy vs. cost"
                      + ("  [PRELIMINARY]" if missing else ""))
            plt.margins(x=0.22, y=0.12)  # headroom so point labels aren't clipped
            plt.grid(alpha=0.3, zorder=0)
            plt.legend(title="Category")
            plt.tight_layout()
            plt.savefig(CHARTS / "accuracy_vs_cost.png", dpi=200)
            plt.close()

        piv = df[df.full_n].pivot(index="model", columns="condition",
                                  values="accuracy")
        if piv.notna().any().any():
            colors = [OPEN_COLOR if m in open_models else PROP_COLOR
                      for m in piv.index]
            piv = piv.rename(index=short_name)
            ax = piv.plot.bar(rot=20, figsize=(8, 5))
            for lbl, col in zip(ax.get_xticklabels(), colors):
                lbl.set_color(col)
            ax.set_ylabel("Accuracy")
            ax.set_title("Zero-shot vs few-shot accuracy by model"
                         + ("  [PRELIMINARY]" if missing else "") + "\n"
                         "(blue = open-weights, red = proprietary)")
            plt.tight_layout()
            plt.savefig(CHARTS / "zs_vs_fs.png", dpi=200)
            plt.close()

    # Two summarization corpora. TweetSumm is the PRIMARY result: it is genuine
    # customer-support dialogue, the domain this paper is about. DialogSum is
    # retained as a secondary cross-domain robustness check — it is
    # general-purpose daily conversation, so agreement between the two is
    # evidence the finding is not an artifact of one corpus.
    for cfgd in SUMM_DATASETS:
        s = cfgd["data"]
        if not s:
            continue
        primary = cfgd["primary"]
        heading = "## Summarization — {} ({})\n".format(
            s.get("dataset"), "PRIMARY, in-domain customer support" if primary
            else "SECONDARY, cross-domain robustness check")
        lines.append("\n" + heading)
        if not primary:
            lines.append(
                "_DialogSum is general-domain daily conversation, not contact "
                "center data. It is reported as a robustness check on whether "
                "the TweetSumm ordering holds out of domain — not as the "
                "headline summarization result._\n")
        rows2 = []
        for m, v in s["models"].items():
            rows2.append({"model": m, "category": category_of(m, open_models),
                          **v, **scl.get((m, cfgd["key"]), {}),
                          **lp.get((m, cfgd["key"]), {})})
        df2 = pd.DataFrame(rows2).sort_values(["category", "model"])
        df2 = ensure_cols(df2, ["rouge1", "rouge2", "rougeL",
                                "list_cost_per_1k", "billed_cost_per_1k",
                                "median_latency_s", "n_calls", "priced_frac",
                                "mean_input_tokens", "mean_output_tokens"])
        df2["full_n"] = df2["n_scored"] >= cfg["summ"]["n_dialogues"]
        sub2 = df2[~df2.full_n]
        lines.append(df2[df2.full_n].drop(columns=["full_n"])
                     .to_markdown(index=False) + "\n")
        if len(sub2):
            lines.append("\n### Supplementary: reduced-sample models "
                         "(EXCLUDED from the compliance gap)\n")
            lines.append(sub2.drop(columns=["full_n"]).to_markdown(index=False) + "\n")
        gap_rows += compliance_gap_rows(
            df2, ["rougeL", "rouge1", "rouge2"],
            "summarization ({}{})".format(
                s.get("dataset"), "" if primary else ", secondary"))

        rl = df2[df2.full_n].dropna(subset=["list_cost_per_1k", "rougeL"])
        if len(rl):
            plt.figure(figsize=(7.5, 5.5))
            for cat, color in [("open-weights", OPEN_COLOR),
                               ("proprietary", PROP_COLOR)]:
                c = rl[rl.category == cat]
                if len(c):
                    plt.scatter(c.list_cost_per_1k, c.rougeL, s=90, c=color,
                                label=cat, edgecolors="black", linewidths=0.5,
                                zorder=3)
            for _, r in rl.iterrows():
                plt.annotate(short_name(r.model),
                             (r.list_cost_per_1k, r.rougeL), fontsize=8,
                             xytext=(6, 6), textcoords="offset points")
            plt.xlabel("Cost per 1,000 summaries (USD, hosted API list price)")
            plt.ylabel("ROUGE-L")
            plt.title("Summarization ({}): ROUGE-L vs. cost".format(s.get("dataset"))
                      + ("  [PRELIMINARY]" if missing else ""))
            plt.margins(x=0.22, y=0.12)  # headroom so point labels aren't clipped
            plt.grid(alpha=0.3, zorder=0)
            plt.legend(title="Category")
            plt.tight_layout()
            plt.savefig(CHARTS / cfgd["chart"], dpi=200)
            plt.close()

    if gap_rows:
        lines.append("\n## Compliance gap: best proprietary vs. best open-weights\n")
        lines.append(
            "`abs_gap` = proprietary - open (positive favors proprietary). "
            "`rel_gap_pct` = abs_gap / proprietary score, in percent. "
            "Cost columns are hosted API list price (see below).\n")
        lines.append(
            "> **These gaps are an UPPER BOUND on the true full-precision "
            "gap.** The 70B was served FP8-quantized (Together's non-quantized "
            "build is not serverless on this account), and quantization can "
            "only depress the open-weights score. Since "
            "`gap = proprietary - open`, a depressed `open` inflates the gap: "
            "`gap_measured >= gap_true`. FP8 can overstate the gap, never "
            "understate it. The small-model skew on the open side (NOTES.md "
            "§3) pushes the same direction, so the two caveats compound. Quote "
            "these figures with the upper-bound qualifier attached.\n")
        lines.append(pd.DataFrame(gap_rows).to_markdown(index=False) + "\n")
    else:
        lines.append("\n## Compliance gap\n\nNot computable: needs at least one "
                     "scored open-weights model and one scored proprietary "
                     "model.\n")

    lines.append("\n## Cost basis\n")
    lines.append(
        "`list_cost_per_1k` is **hosted API list price**, not what this study "
        "was billed. Four of the five models ran on free promotional tiers, so "
        "`billed_cost_per_1k` is ~$0 for them and would make the cost axis an "
        "artifact of promotions rather than an economic comparison. It is also "
        "**not** self-hosting TCO (GPU capital/rental, engineering time, idle "
        "capacity) — that belongs in the paper's limitations.\n")
    lines.append(
        "Token counts are **estimated**: the providers' `usage` fields were not "
        "persisted, so prompts were reconstructed exactly as the runners built "
        "them and tokenized with `litellm.token_counter`, along with the stored "
        "output text. For non-OpenAI models this is an approximation (a few "
        "percent), excludes chat-template scaffolding, and is applied "
        "identically to every model — sound for comparison, quote absolute "
        "values as estimates. See `src/list_price.py` and NOTES.md §9.\n")
    lines.append(list_price.price_table_markdown() + "\n")

    (RES / "tables.md").write_text("\n".join(lines))
    print(f"Wrote {RES/'tables.md'} and charts to {CHARTS}/")
    if missing:
        print(f"NOTE: report is PRELIMINARY — {len(missing)} model(s) incomplete")


if __name__ == "__main__":
    main()
