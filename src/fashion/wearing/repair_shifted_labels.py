"""Repair wearing labels shifted by the old run_wearing_model re-nesting bug.

prepare_batch drops candidate characters that fall outside the excerpt, so
WearingBert predicted one label per in-bounds character. The old is_wearing then
re-nested those predictions by the count of *all* characters, and label_batch
zipped them onto all characters. run_wearing_model sends a whole file as one batch,
so every dropped character shifted the labels of every later mention in the file,
and the predictions ran out before the file's last mentions (which got no
"wearing" key).

The predictions themselves survive intact: read in file order, the stored labels
are exactly the flat prediction list. This script rebuilds the labels from that
list without rerunning the model, giving each in-bounds character its own
prediction and every out-of-bounds character False, as the fixed label_batch does.

Each file is checked for the bug's signature first: the characters with a
"wearing" key are exactly as many as the in-bounds characters, and all come first.
Files without dropped characters were never shifted and files that already have a
label on every character with none out of bounds labeled True look repaired; both
are copied unchanged. Any other file is reported and not written.

Usage:
    python -m fashion.wearing.repair_shifted_labels --wearing_dir DIR --output_dir DIR
"""

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fashion.wearing.eval import in_bounds_characters

EXAMPLE_CONTEXT = 150  # characters of excerpt shown on either side of the term


def scan(path: Path) -> dict:
    """First pass: the stored labels in file order and the per-mention counts."""
    stored = []  # True / False / None (no key), one per character
    n_mentions = n_dropped_mentions = n_in_bounds = oob_true = 0
    with open(path) as f:
        for line in f:
            datum = json.loads(line)
            in_bounds = in_bounds_characters(datum)
            n_mentions += 1
            n_in_bounds += sum(in_bounds)
            n_dropped_mentions += not all(in_bounds)
            for char, is_in_bounds in zip(datum["characters"], in_bounds):
                stored.append(char.get("wearing"))
                oob_true += not is_in_bounds and char.get("wearing") is True
    n_keyed = sum(label is not None for label in stored)
    return {
        "stored": stored,
        "n_mentions": n_mentions,
        "n_dropped_mentions": n_dropped_mentions,
        "n_characters": len(stored),
        "n_in_bounds": n_in_bounds,
        "n_keyed": n_keyed,
        "keys_first": all(label is not None for label in stored[:n_keyed]),
        "oob_true": oob_true,
    }


def classify(stats: dict) -> str:
    if stats["n_in_bounds"] == stats["n_characters"]:
        return "unshifted"
    if stats["n_keyed"] == stats["n_in_bounds"] and stats["keys_first"]:
        return "shifted"
    if stats["n_keyed"] == stats["n_characters"] and stats["oob_true"] == 0:
        return "repaired"
    return "unknown"


def wearers(datum: dict, labels: list[bool | None]) -> str:
    """The characters labeled wearing, as text_coref@distance."""
    found = [
        f"{char['text']}_{char['coref']}@{char['distance']}"
        for char, label in zip(datum["characters"], labels)
        if label
    ]
    return ", ".join(found) or "(none)"


def format_example(datum: dict, old: list[bool | None], new: list[bool]) -> str:
    """The excerpt around the fashion term, with the wearers before and after."""
    start = datum["fashion_start_idx"] - datum["excerpt_start"]
    end = datum["fashion_end_idx"] - datum["excerpt_start"]
    text = datum["excerpt_text"]
    context = (
        text[max(0, start - EXAMPLE_CONTEXT) : start]
        + f"[{text[start:end]}]"
        + text[end : end + EXAMPLE_CONTEXT]
    )
    context = " ".join(context.split())
    return (
        f"  {datum['book_id']} mention {datum['mention_id']} "
        f"({datum['fashion_term']}):\n"
        f"    ...{context}...\n"
        f"    before: {wearers(datum, old)}\n"
        f"    after:  {wearers(datum, new)}"
    )


def repair(
    path: Path, output_dir: Path, predictions: list[bool], num_examples: int
) -> dict:
    """Second pass: rewrite each mention with its own predictions, and sample
    num_examples mentions whose wearer corefs changed (reservoir sampling)."""
    output_path = output_dir / path.name
    tmp_path = output_path.with_suffix(".ndjson.tmp")
    position = changed_mentions = changed_corefs = 0
    rng = random.Random(path.name)
    examples = []
    with open(path) as f, open(tmp_path, "w") as out:
        for line in f:
            datum = json.loads(line)
            old = [char.get("wearing") for char in datum["characters"]]
            new = []
            for char, is_in_bounds in zip(
                datum["characters"], in_bounds_characters(datum)
            ):
                if is_in_bounds:
                    label = predictions[position]
                    position += 1
                else:
                    label = False
                new.append(label)
                char["wearing"] = label
            changed_mentions += old != new
            corefs = [
                {char["coref"] for char, w in zip(datum["characters"], labels) if w}
                for labels in (old, new)
            ]
            if corefs[0] != corefs[1]:
                changed_corefs += 1
                if len(examples) < num_examples:
                    examples.append(format_example(datum, old, new))
                elif (slot := rng.randrange(changed_corefs)) < num_examples:
                    examples[slot] = format_example(datum, old, new)
            out.write(json.dumps(datum) + "\n")
    if position != len(predictions):
        tmp_path.unlink()
        raise RuntimeError(f"{path}: used {position} of {len(predictions)} predictions")
    os.replace(tmp_path, output_path)
    return {
        "changed_mentions": changed_mentions,
        "changed_corefs": changed_corefs,
        "examples": examples,
    }


def process_file(path: Path, output_dir: Path, num_examples: int) -> str:
    stats = scan(path)
    kind = classify(stats)
    summary = (
        f"{path.name}: {stats['n_mentions']:,} mentions, "
        f"{stats['n_dropped_mentions']:,} with out-of-bounds characters "
        f"({stats['n_characters'] - stats['n_in_bounds']:,} characters)"
    )
    if kind in ("unshifted", "repaired"):
        shutil.copyfile(path, output_dir / path.name)
        return f"{summary}; {kind}, copied unchanged"
    if kind == "unknown":
        return (
            f"{summary}; NOT WRITTEN: {stats['n_keyed']:,} labeled characters, "
            f"{stats['n_in_bounds']:,} in bounds, {stats['n_characters']:,} total, "
            f"labeled characters first: {stats['keys_first']}, out-of-bounds "
            f"labeled True: {stats['oob_true']:,}"
        )
    # stored labels are False for both skipped and negative predictions, as before
    predictions = [bool(label) for label in stats["stored"][: stats["n_in_bounds"]]]
    result = repair(path, output_dir, predictions, num_examples)
    header = (
        f"{summary}; repaired, {result['changed_mentions']:,} mentions "
        f"({result['changed_mentions'] / stats['n_mentions']:.1%}) changed, "
        f"{result['changed_corefs']:,} "
        f"({result['changed_corefs'] / stats['n_mentions']:.1%}) with different "
        "wearer corefs"
    )
    lines = [header]
    if result["examples"]:
        lines.append("  sample of mentions whose wearer corefs changed:")
        lines += result["examples"]
    return "\n".join(lines)


def main(wearing_dir: Path, output_dir: Path, workers: int, num_examples: int):
    if wearing_dir.resolve() == output_dir.resolve():
        raise SystemExit("--output_dir must differ from --wearing_dir")
    files = sorted(wearing_dir.glob("*.ndjson"))
    if not files:
        raise SystemExit(f"No *.ndjson files in {wearing_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(process_file, path, output_dir, num_examples) for path in files
        ]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wearing_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Files repaired in parallel (each holds its labels in memory).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Mentions per repaired file to print, sampled from those whose wearer "
        "corefs changed: the text around the fashion term and the wearers (as "
        "text_coref@distance) before and after.",
    )
    args = parser.parse_args()

    main(args.wearing_dir, args.output_dir, args.workers, args.examples)
