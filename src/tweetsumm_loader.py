"""Reconstruct TweetSumm dialogues from tweet IDs + the Kaggle twcs.csv.

LICENSING — read before changing anything here
----------------------------------------------
TweetSumm (https://github.com/guyfe/Tweetsumm) distributes only tweet *IDs*,
sentence offsets and human annotations. The tweet TEXT lives in Kaggle's
"Customer Support on Twitter" dataset (`twcs.csv`) and is deliberately NOT
redistributed by TweetSumm. Their dataset is released under
**CDLA-Sharing-1.0**; the repo's own code carries CC0-1.0.

Consequently this project does not commit `data/twcs.csv`, the vendored repo,
or any reconstructed dialogue text. Only this loader and the tweet IDs are
public, which keeps text access routed through Kaggle as the license intends.
`.gitignore` enforces it.

RECONSTRUCTION
--------------
The upstream `TweetSumProcessor` does the reconstruction correctly (offset
slicing, agent/customer direction), so we reuse it verbatim rather than
reimplementing and risking a subtle mismatch. But its constructor loads all
~3M rows of twcs.csv into a dict, which is gigabytes of resident memory for a
split that references only ~1k tweets. We therefore build the same
`tweet_id_to_content` mapping ourselves from a single streaming pass, keeping
only the IDs the split actually needs, and hand it to an instance created via
`__new__`. All parsing logic downstream is upstream's, untouched.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "tweetsumm"
DATA_FILES = VENDOR / "tweet_sum_data_files"
TWCS = ROOT / "data" / "twcs.csv"

SPLITS = {
    "test": "final_test_tweetsum.jsonl",
    "valid": "final_valid_tweetsum.jsonl",
    "train": "final_train_tweetsum.jsonl",
}

# twcs.csv contains tweets with embedded newlines and very long fields.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _load_processor(needed_ids: set[str]):
    """Upstream TweetSumProcessor with only `needed_ids` resident in memory."""
    if not VENDOR.exists():
        raise FileNotFoundError(
            f"{VENDOR} missing — clone https://github.com/guyfe/Tweetsumm there")
    if not TWCS.exists():
        raise FileNotFoundError(
            f"{TWCS} missing — download twcs.csv from the Kaggle 'Customer "
            "Support on Twitter' dataset (not redistributable here)")

    sys.path.insert(0, str(VENDOR))
    from tweet_sum_processor import TweetSumProcessor

    mapping = {}
    with open(TWCS, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 5:
                continue
            tid = str(row[0])
            if tid in needed_ids:
                mapping[tid] = (row[2], row[4])  # (inbound, text)
                if len(mapping) == len(needed_ids):
                    break

    proc = TweetSumProcessor.__new__(TweetSumProcessor)
    proc.tweet_id_to_content = mapping
    return proc, mapping


def load_split(split: str = "test"):
    """-> (list of {id, dialogue, reference}, stats dict). No sampling here."""
    path = DATA_FILES / SPLITS[split]
    lines = [l for l in path.read_text().splitlines() if l.strip()]

    needed = set()
    for l in lines:
        d = json.loads(l)
        for t in d["tweet_ids_sentence_offset"]:
            needed.add(str(t["tweet_id"]))

    proc, mapping = _load_processor(needed)
    missing = needed - set(mapping)

    # Drop dialogues referencing tweets absent from this twcs.csv snapshot
    # rather than emitting a partial conversation.
    usable_lines, skipped = [], 0
    for l in lines:
        d = json.loads(l)
        ids = {str(t["tweet_id"]) for t in d["tweet_ids_sentence_offset"]}
        if ids & missing:
            skipped += 1
        else:
            usable_lines.append(l)

    dialogs = proc.get_dialog_with_summaries(usable_lines)

    items = []
    for dws in dialogs:
        dialog = dws.get_dialog()
        turns = []
        for turn in dialog.get_turns():
            speaker = "Agent" if turn.is_agent() else "Customer"
            text = " ".join(s.strip() for s in turn.get_sentences() if s.strip())
            if text:
                turns.append(f"{speaker}: {text}")
        abstractive = dws.get_abstractive_summaries()
        if not turns or not abstractive:
            continue
        # Each annotation is a list of sentences; join each into one summary.
        # ALL annotations are kept: `reference` (the first) preserves the
        # original single-reference scoring for continuity with DialogSum, and
        # `references` enables multi-reference ROUGE (max over references),
        # which is the standard convention when a corpus ships several human
        # summaries per item and is the fairer measurement.
        refs = [" ".join(s.strip() for s in ann if s.strip())
                for ann in abstractive]
        refs = [r for r in refs if r]
        if not refs:
            continue
        items.append({
            "id": f"tweetsumm_{dialog.get_dialog_id()}",
            "dialogue": "\n".join(turns),
            "reference": refs[0],
            "references": refs,
            "n_abstractive_annotations": len(refs),
        })

    stats = {
        "split": split,
        "lines_in_split": len(lines),
        "tweet_ids_referenced": len(needed),
        "tweet_ids_found_in_twcs": len(mapping),
        "dialogues_skipped_missing_tweets": skipped,
        "dialogues_usable": len(items),
    }
    return items, stats


def sample(n: int = 100, split: str = "test", seed: int = 42):
    """Seeded sample, matching how the other runners sample (shuffle then slice)."""
    items, stats = load_split(split)
    rng = random.Random(seed)
    rng.shuffle(items)
    stats["n_sampled"] = min(n, len(items))
    return items[:n], stats


if __name__ == "__main__":
    items, stats = sample(100)
    print(json.dumps(stats, indent=2))
    if items:
        it = items[0]
        print("\n--- example ---")
        print("id:", it["id"])
        print("dialogue (first 400 chars):")
        print(it["dialogue"][:400])
        print("\nreference:", it["reference"])
