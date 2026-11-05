"""Per-decade prevalence of fashion adjectives, terms, and adjective + term pairs.

Builds the data behind the interactive *Adjective Trends* page in ``visualize/``.
Starts from the co-occurrence code in ``notebooks/cooccurrences.ipynb``
(``adjectives_fashion`` × ``term_stem``, adjectives NFKC-normalized and
lowercased) and splits every count by decade and by character gender, so the
front-end can compute any of several prevalence measures on the fly.

The unit is the fashion *mention*. An adjective is counted once per mention it
describes (repeats within a mention are collapsed), so "adjective count" means
"mentions described with that adjective". Alongside each mention count we store
the number of distinct books it comes from, so the page can also show the share
of books, which keeps a handful of verbose books from dominating a decade.

It also writes preset adjective groups: WordNet attribute categories from
``adjective_categories.py``, which the page offers as editable starting points
for grouping adjectives.

Example sentences are written to small per-adjective and per-term files, loaded
lazily by the page when the user asks for examples.

Output: data/analysis/adjective_trends/
    trends.json                 counts (see the analysis README for the schema)
    examples/adjs/{id}.json     examples per adjective, and per (adjective, term)
    examples/terms/{id}.json    examples per term
"""

import argparse
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fashion.analysis.adjective_categories import get_attribute_categories
from fashion.paths import DATA_DIR

EXPERIMENT = "adjective_trends"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
GENDERS = ("all", "male", "female")
GENDER_CODES = {"male": "m", "female": "f"}

# Letters only, optionally joined by hyphens or apostrophes. Drops OCR and
# parser debris such as "■mart", "'anonymous", and elisions like "an'".
ADJ_PATTERN = re.compile(r"^[^\W\d_]+(?:[-'’][^\W\d_]+)*$")

# Characters of context kept on each side of the fashion term in an example.
CONTEXT_CHARS = 220


def clean_adj(adj) -> str | None:
    """Normalize an adjective as in the notebook; None if it is not a word."""
    if adj is None:
        return None
    adj = unicodedata.normalize("NFKC", str(adj)).lower().strip()
    return adj if ADJ_PATTERN.match(adj) else None


def clean_adjs(value) -> list[str]:
    """Clean a list cell of adjectives, keeping the first of any repeats."""
    if not isinstance(value, (list, np.ndarray)):
        return []
    return list(dict.fromkeys(a for a in map(clean_adj, value) if a is not None))


def _squash(text: str) -> str:
    """Collapse whitespace and fix ligatures, matching the other exports."""
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text)


def snippet(sentence: str, start: int, end: int) -> tuple[str, int, int]:
    """Clean a sentence around the term, returning it and the term's offsets.

    ``start``/``end`` are offsets of the term within the raw sentence. Each side
    is trimmed to ``CONTEXT_CHARS`` at a word boundary.
    """
    pre = _squash(sentence[:start]).lstrip()
    mid = _squash(sentence[start:end])
    post = _squash(sentence[end:]).rstrip()
    if len(pre) > CONTEXT_CHARS:
        cut = pre[-CONTEXT_CHARS:]
        pre = "…" + cut[cut.find(" ") + 1 :] if " " in cut else "…" + cut
    if len(post) > CONTEXT_CHARS:
        cut = post[:CONTEXT_CHARS]
        post = (cut[: cut.rfind(" ")] if " " in cut else cut) + "…"
    return pre + mid + post, len(pre), len(pre) + len(mid)


def load(input_path: Path) -> pd.DataFrame:
    """Load mentions with cleaned adjectives, a gender bucket, and a decade."""
    columns = [
        "term",
        "term_stem",
        "sentence",
        "start_idx",
        "end_idx",
        "adjectives_fashion",
        "gender_dep_pron",
        "decade",
    ]
    df = pd.read_parquet(input_path, columns=columns).reset_index()
    df = df.dropna(subset=["decade", "term_stem"]).reset_index(drop=True)
    df["decade"] = (df["decade"] * 10).astype(int)
    df["term_stem"] = df["term_stem"].astype(str)
    # Only male/female are split out; everything else counts toward "all" only.
    df["gender"] = df["gender_dep_pron"].where(
        df["gender_dep_pron"].isin(["male", "female"])
    )
    df["adjs"] = [
        clean_adjs(v) for v in tqdm(df["adjectives_fashion"], desc="Adjectives")
    ]
    return df.drop(columns=["adjectives_fashion", "gender_dep_pron"])


def explode_adjs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (mention, adjective); ``row`` points back into ``df``."""
    ex = df[["adjs", "term_stem", "decade", "book_id", "gender"]].explode("adjs")
    ex = ex.dropna(subset=["adjs"]).rename(columns={"adjs": "adj"})
    return ex.rename_axis("row").reset_index()


def count_table(frame: pd.DataFrame, keys: list[str], decades: list[int]) -> dict:
    """Mention and distinct-book counts per (key, decade) for each gender slice.

    Returns ``{gender: (mentions, books)}``, each a dict mapping a key (a tuple
    when there are several ``keys``) to its list of per-decade counts.
    """
    tables = {}
    for gender in GENDERS:
        sub = frame if gender == "all" else frame[frame["gender"] == gender]
        grouped = sub.groupby(keys + ["decade"])
        tables[gender] = tuple(
            dict(zip(wide.index, wide.to_numpy().tolist()))
            for wide in (
                counts.unstack("decade", fill_value=0).reindex(
                    columns=decades, fill_value=0
                )
                for counts in (grouped.size(), grouped["book_id"].nunique())
            )
        )
    return tables


def series_record(tables: dict, key, n_decades: int) -> dict:
    """``{"n": {gender: [...]}, "b": {gender: [...]}}`` for one key."""
    zeros = [0] * n_decades
    return {
        field: {gender: pair[i].get(key, zeros) for gender, pair in tables.items()}
        for i, field in enumerate(("n", "b"))
    }


def totals_record(df: pd.DataFrame, decades: list[int]) -> dict:
    """Per-decade denominators for each gender slice."""
    totals = {}
    df = df.assign(described=df["adjs"].map(bool), adj_tokens=df["adjs"].map(len))
    for gender in GENDERS:
        sub = df if gender == "all" else df[df["gender"] == gender]
        grouped = sub.groupby("decade")
        frame = pd.DataFrame(
            {
                "mentions": grouped.size(),
                "books": grouped["book_id"].nunique(),
                "described": grouped["described"].sum(),
                "adj_tokens": grouped["adj_tokens"].sum(),
            }
        ).reindex(decades, fill_value=0)
        totals[gender] = {col: [int(v) for v in frame[col]] for col in frame.columns}
    return totals


def sample_rows(
    frame: pd.DataFrame, keys: list[str], k: int, k_gender: int, seed: int
) -> pd.DataFrame:
    """Pick example rows per (key, decade), spreading them across books.

    Takes up to ``k`` rows per (key, decade) regardless of gender, plus up to
    ``k_gender`` per (key, decade, gender) for male and female, so the page has
    examples to show when it is filtered to one gender. Within a group, rows
    from books not yet sampled come first.
    """
    rng = np.random.default_rng(seed)
    cols = keys + ["decade", "book_id", "gender", "row"]
    frame = frame[cols].assign(_r=rng.random(len(frame))).sort_values("_r")
    frame["_book_rank"] = frame.groupby(keys + ["decade", "book_id"]).cumcount()
    frame = frame.sort_values(["_book_rank", "_r"])
    picked = [frame.groupby(keys + ["decade"]).head(k)]
    if k_gender > 0:
        gendered = frame.dropna(subset=["gender"])
        picked.append(gendered.groupby(keys + ["decade", "gender"]).head(k_gender))
    return pd.concat(picked).drop_duplicates(subset=keys + ["row"])


def example_record(row) -> dict:
    """Compact example payload for one mention.

    Title, author, and year are looked up by ``bk`` in ``books.py``'s output.
    """
    text, start, end = snippet(row.sentence, int(row.start_idx), int(row.end_idx))
    return {
        "s": text,
        "h": [start, end],
        "a": row.adjs,
        "bk": row.book_id,
        "g": GENDER_CODES.get(row.gender),
    }


def group_examples(rows: pd.DataFrame, keys: list[str], records: dict) -> dict:
    """Nest example records as ``{key..., decade: [records]}``, newest last."""
    nested: dict = {}
    rows = rows.sort_values(keys + ["decade", "row"])
    for values in rows[keys + ["decade", "row"]].itertuples(index=False):
        *key_parts, decade, row = values
        node = nested
        for part in key_parts:
            node = node.setdefault(part, {})
        node.setdefault(str(decade), []).append(records[row])
    return nested


def category_presets(adjs: list[str], adj_totals: pd.Series, min_members: int) -> list:
    """WordNet attribute categories as preset adjective groups.

    Uses ``get_attribute_categories`` from ``adjective_categories.py`` *without*
    context, so an adjective joins every category any of its senses maps to. The
    page shows each group's members so strays can be removed. That script's
    per-mention Lesk disambiguation isn't used here: on fashion adjectives it puts
    most uses of "grey" under *confederacy*, and leaves half of all uses
    (including most of "white", "black", and "long") with no category.

    Returns ``[{id, name, definition, members}]`` with members ordered by use,
    groups ordered by their members' total use. Empty if WordNet is missing.
    """
    try:
        from nltk.corpus import wordnet as wn

        wn.ensure_loaded()
    except LookupError:
        print(
            "WordNet data not found; skipping preset groups (nltk.download('wordnet'))"
        )
        return []
    members: dict[str, list[str]] = {}
    for adj in sorted(adjs, key=lambda a: -adj_totals[a]):
        for synset in {s.name() for s in get_attribute_categories(term=adj)}:
            members.setdefault(synset, []).append(adj)
    members = {k: v for k, v in members.items() if len(v) >= min_members}
    lemma_uses = Counter(name.split(".")[0] for name in members)
    presets = []
    for synset, adjs_in in members.items():
        lemma, _, sense = synset.split(".")
        name = lemma.replace("_", " ")
        if lemma_uses[lemma] > 1:
            name = f"{name} {int(sense)}"
        presets.append(
            {
                "id": synset,
                "name": name,
                "definition": wn.synset(synset).definition(),
                "members": adjs_in,
                "_uses": int(adj_totals[adjs_in].sum()),
            }
        )
    presets.sort(key=lambda p: -p["_uses"])
    for preset in presets:
        del preset["_uses"]
    return presets


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        # allow_nan=False so a stray NaN fails here rather than in the browser.
        json.dump(
            payload, file, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Override the input parquet path."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where to write the output."
    )
    parser.add_argument(
        "--min-adj-count",
        type=int,
        default=20,
        help="Keep adjectives describing at least this many mentions (default: 20).",
    )
    parser.add_argument(
        "--min-term-count",
        type=int,
        default=20,
        help="Keep term stems with at least this many mentions (default: 20).",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=3,
        help="Keep (adjective, term) pairs with at least this many mentions "
        "(default: 3). Low, since adjective groups add up their members' pairs.",
    )
    parser.add_argument(
        "--min-pair-example-count",
        type=int,
        default=10,
        help="Only sample examples for pairs with at least this many mentions "
        "(default: 10), to keep the example files small.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Smallest preset adjective group to keep (default: 2).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Example sentences per (adjective or term, decade) (default: 5).",
    )
    parser.add_argument(
        "--pair-examples",
        type=int,
        default=3,
        help="Example sentences per (adjective, term, decade) (default: 3).",
    )
    parser.add_argument(
        "--gender-examples",
        type=int,
        default=2,
        help="Extra examples per (series, decade) for each of male and female, "
        "so gender-filtered views have examples (default: 2; 0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )
    print(f"Loading {input_path}")
    df = load(input_path)
    decades = sorted(df["decade"].unique().tolist())
    print(
        f"{len(df):,} mentions across {df['book_id'].nunique():,} books, decades {decades}"
    )

    ex = explode_adjs(df)
    adj_totals = ex["adj"].value_counts()
    term_totals = df["term_stem"].value_counts()
    adjs = sorted(adj_totals[adj_totals >= args.min_adj_count].index)
    terms = sorted(term_totals[term_totals >= args.min_term_count].index)
    ex = ex[ex["adj"].isin(adjs)]
    pair_totals = ex[ex["term_stem"].isin(terms)].groupby(["adj", "term_stem"]).size()
    pairs = pair_totals[pair_totals >= args.min_pair_count].index
    print(
        f"Keeping {len(adjs):,} adjectives, {len(terms):,} terms, {len(pairs):,} pairs"
    )

    # Display label for a stem: its most common surface form.
    labels = (
        df.assign(term=df["term"].str.lower())
        .groupby("term_stem")["term"]
        .agg(lambda s: s.value_counts().index[0])
    )

    print("Counting")
    term_tables = count_table(df, ["term_stem"], decades)
    adj_tables = count_table(ex, ["adj"], decades)
    pair_ex = ex.set_index(["adj", "term_stem"])
    pair_ex = pair_ex[pair_ex.index.isin(pairs)].reset_index()
    pair_tables = count_table(pair_ex, ["adj", "term_stem"], decades)

    pair_records: dict[str, dict] = {}
    for adj, stem in tqdm(pairs, desc="Pairs"):
        pair_records.setdefault(adj, {})[stem] = series_record(
            pair_tables, (adj, stem), len(decades)
        )

    trends = {
        "setup": {
            "input": input_path.name,
            "unit": "mention",
            "gender_column": "gender_dep_pron",
            "min_adj_count": args.min_adj_count,
            "min_term_count": args.min_term_count,
            "min_pair_count": args.min_pair_count,
            "min_pair_example_count": args.min_pair_example_count,
            "examples": args.examples,
            "pair_examples": args.pair_examples,
            "gender_examples": args.gender_examples,
        },
        "decades": decades,
        "genders": list(GENDERS),
        "totals": totals_record(df, decades),
        "terms": {
            stem: {
                "id": i,
                "label": labels[stem],
                **series_record(term_tables, stem, len(decades)),
            }
            for i, stem in enumerate(terms)
        },
        "adjs": {
            adj: {"id": i, **series_record(adj_tables, adj, len(decades))}
            for i, adj in enumerate(adjs)
        },
        "pairs": pair_records,
        "groups": category_presets(adjs, adj_totals, args.min_group_size),
    }
    print(f"{len(trends['groups']):,} preset adjective groups")

    out_dir = args.output_dir
    examples_dir = out_dir / "examples"
    # Example files are keyed by id, which shifts between runs; clear stale ones.
    if examples_dir.exists():
        shutil.rmtree(examples_dir)
    (examples_dir / "adjs").mkdir(parents=True)
    (examples_dir / "terms").mkdir(parents=True)
    write_json(out_dir / "trends.json", trends)
    print(f"  → {out_dir / 'trends.json'}")

    print("Sampling examples")
    mentions = df[df["term_stem"].isin(terms)].rename_axis("row").reset_index()
    term_rows = sample_rows(
        mentions, ["term_stem"], args.examples, args.gender_examples, args.seed
    )
    adj_rows = sample_rows(ex, ["adj"], args.examples, args.gender_examples, args.seed)
    example_pairs = pair_totals[pair_totals >= args.min_pair_example_count].index
    pair_rows = sample_rows(
        pair_ex[pair_ex.set_index(["adj", "term_stem"]).index.isin(example_pairs)],
        ["adj", "term_stem"],
        args.pair_examples,
        args.gender_examples,
        args.seed,
    )

    needed = pd.Index(term_rows["row"]).union(adj_rows["row"]).union(pair_rows["row"])
    records = {
        row.Index: example_record(row)
        for row in tqdm(df.loc[needed].itertuples(), total=len(needed), desc="Examples")
    }

    term_examples = group_examples(term_rows, ["term_stem"], records)
    for stem in tqdm(terms, desc="Writing term examples"):
        payload = {"term": stem, "examples": term_examples.get(stem, {})}
        write_json(
            examples_dir / "terms" / f"{trends['terms'][stem]['id']}.json", payload
        )

    adj_examples = group_examples(adj_rows, ["adj"], records)
    pair_examples = group_examples(pair_rows, ["adj", "term_stem"], records)
    for adj in tqdm(adjs, desc="Writing adjective examples"):
        payload = {
            "adj": adj,
            "examples": adj_examples.get(adj, {}),
            "pairs": pair_examples.get(adj, {}),
        }
        write_json(examples_dir / "adjs" / f"{trends['adjs'][adj]['id']}.json", payload)

    total_mb = sum(p.stat().st_size for p in out_dir.rglob("*.json")) / 1e6
    print(f"Wrote {len(adjs):,} adjective and {len(terms):,} term example files")
    print(f"  → {out_dir} ({total_mb:.1f} MB total)")


if __name__ == "__main__":
    main()
