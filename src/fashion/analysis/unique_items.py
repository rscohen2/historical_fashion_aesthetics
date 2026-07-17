"""Per-character "unique items" fashion profiles for interactive exploration.

Mirrors the notebook's *Unique items* section (``top_chars_w_fashion_mention``
and friends): for each character (``book_id``, ``character_id``) we measure how
"fashion-y" they are along three axes and package everything the front-end needs
to render explorable cards and a scatter plot.

Metrics per character:

1. ``num_mentions``  — total fashion mentions (count of ``term``)
2. ``num_distinct``  — distinct fashion items worn (nunique ``term``)
3. ``ratio``         — mentions / distinct (wears the same item repeatedly)

For a bounded pool of the most fashion-y characters we also collect the top
clothing items, fashion adjectives, character adjectives, and the context
sentences for every item, so a card can be expanded down to the passage level.

Output: data/analysis/unique_items/unique_items.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from fashion.paths import DATA_DIR

EXPERIMENT = "unique_items"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
DECADES = list(range(180, 192))

# The three rankings surfaced as card lists, keyed by the metric column they
# sort on (descending). Mirrors the notebook's three ``top_chars_*`` functions.
RANKINGS = {
    "most_mentions": "num_mentions",
    "most_distinct": "num_distinct",
    "most_repeated": "ratio",
}

# Narrator / non-character narration. BookNLP assigns character_id 0 to text not
# linked to a named character (the notebook's ``include_narr`` toggle keys off
# ``not character_id``). We keep these rows and let the interface filter them.
NARRATOR_ID = 0


def _clean(text: str) -> str:
    """Collapse whitespace and fix ligatures, matching the log-odds export."""
    text = text.replace("\n", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text).strip()


def _iter_adjs(value):
    """Yield lowercased, non-null adjectives from a list/array cell."""
    if isinstance(value, (list, np.ndarray)):
        for adj in value:
            if adj is not None:
                yield str(adj).lower()


def char_key(book_id: str, character_id) -> str:
    """Stable string id for a (book, character) pair used as a JSON key."""
    return f"{book_id}::{int(character_id)}"


def _clean_meta(value):
    """Coerce a metadata cell to a clean string or None.

    Missing cells in the title metadata come back from pandas as float NaN,
    which ``json.dump`` would write as bare ``NaN`` (invalid JSON that the
    browser's ``fetch().json()`` rejects). Normalise them to ``None``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return _clean(str(value))


def compute_character_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the three fashion-y metrics for every character."""
    grouped = df.groupby(["book_id", "character_id"])["term"]
    stats = pd.DataFrame(
        {
            "num_mentions": grouped.count(),
            "num_distinct": grouped.nunique(),
        }
    )
    stats["ratio"] = stats["num_mentions"] / stats["num_distinct"]
    return stats


def select_pool(
    stats: pd.DataFrame,
    top_cards: int,
    scatter_n: int,
    min_mentions: int,
) -> tuple[dict[str, list[str]], list[str], set[tuple[str, int]]]:
    """Pick which characters need full detail.

    Returns the per-metric ranking lists (as string ids), the scatter list (as
    string ids), and the union set of (book_id, character_id) index pairs that
    need full detail built (every card and every dot has full detail).

    The narrator (``character_id == 0``) dominates the raw counts, so the
    interface lets the user hide narrators. To keep a full top-N in *both* views
    without shipping duplicate lists, each ranking/scatter pool is the union of
    the top-N over all characters and the top-N over non-narrators. The front-end
    then filters by ``is_narrator`` and slices — either view still has enough
    entries because the pool guarantees the top-N of each is present.
    """
    eligible = stats[stats["num_mentions"] >= min_mentions]
    non_narr = eligible[
        eligible.index.get_level_values("character_id") != NARRATOR_ID
    ]

    def _top_union(frame_sort_col: str, n: int) -> pd.Index:
        # Tie-break by the other magnitude metric so ordering is deterministic.
        keys = [frame_sort_col, "num_mentions", "num_distinct"]
        top_all = eligible.sort_values(keys, ascending=False).head(n)
        top_non = non_narr.sort_values(keys, ascending=False).head(n)
        union = eligible.loc[top_all.index.union(top_non.index)]
        return union.sort_values(keys, ascending=False).index

    rankings: dict[str, list[str]] = {}
    detail_pairs: set[tuple[str, int]] = set()
    for name, col in RANKINGS.items():
        idx = _top_union(col, top_cards)
        rankings[name] = [char_key(b, c) for b, c in idx]
        detail_pairs.update((b, int(c)) for b, c in idx)

    scatter_idx = _top_union("num_mentions", scatter_n)
    scatter_ids = [char_key(b, c) for b, c in scatter_idx]
    detail_pairs.update((b, int(c)) for b, c in scatter_idx)

    return rankings, scatter_ids, detail_pairs


def build_character_detail(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    detail_pairs: set[tuple[str, int]],
    metadata: dict,
    top_clothes: int,
    top_adjs: int,
    max_sentences: int,
) -> dict[str, dict]:
    """Build the full card payload for each character in ``detail_pairs``."""
    detail_df = df.set_index(["book_id", "character_id"]).sort_index()
    wanted = pd.MultiIndex.from_tuples(sorted(detail_pairs))
    detail_df = detail_df[detail_df.index.isin(wanted)]

    characters: dict[str, dict] = {}
    for (book_id, character_id), group in tqdm(
        detail_df.groupby(level=[0, 1]), desc="Building character detail"
    ):
        term_counts: Counter = Counter()
        fashion_adj_counts: Counter = Counter()
        char_adjs: set[str] = set()
        sentences_by_term: dict[str, list[str]] = defaultdict(list)

        for row in group.itertuples(index=False):
            term = str(row.term)
            term_counts[term] += 1
            for adj in _iter_adjs(row.adjectives_fashion):
                fashion_adj_counts[adj] += 1
            char_adjs.update(_iter_adjs(row.adjectives_char))
            bucket = sentences_by_term[term]
            if len(bucket) < max_sentences and isinstance(row.sentence, str):
                bucket.append(_clean(row.sentence))

        genders = group["gender_dep_pron"].dropna()
        gender = genders.value_counts().index[0] if len(genders) else None

        book_meta = metadata.get(book_id, {})
        stat = stats.loc[(book_id, character_id)]
        cid = char_key(book_id, character_id)

        top_terms = term_counts.most_common(top_clothes)
        characters[cid] = {
            "id": cid,
            "book_id": book_id,
            "character_id": int(character_id),
            "is_narrator": int(character_id) == NARRATOR_ID,
            "title": _clean_meta(book_meta.get("title")),
            "author": _clean_meta(book_meta.get("author")),
            "gender": gender,
            "num_mentions": int(stat["num_mentions"]),
            "num_distinct": int(stat["num_distinct"]),
            "ratio": round(float(stat["ratio"]), 3),
            "top_clothes": [{"term": t, "count": n} for t, n in top_terms],
            "fashion_adjs": [
                {"adj": a, "count": n}
                for a, n in fashion_adj_counts.most_common(top_adjs)
            ],
            "char_adjs": sorted(char_adjs),
            # Only ship sentences for the items shown as chips on the card.
            "sentences": {
                t: sentences_by_term[t] for t, _ in top_terms if sentences_by_term[t]
            },
        }
    return characters


def load_metadata() -> dict:
    """Load Hathi title metadata keyed by ``book_id`` (docid + ``.clean``)."""
    meta = pd.read_csv(DATA_DIR / "hathimeta" / "titlemeta.tsv", sep="\t")
    meta["book_id"] = meta["docid"] + ".clean"
    return meta.set_index("book_id").to_dict(orient="index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--top-cards",
        type=int,
        default=100,
        help="Number of characters per ranking to expose as cards (default: 100).",
    )
    parser.add_argument(
        "--scatter-n",
        type=int,
        default=800,
        help="Number of characters (by mention count) plotted on the scatter "
        "and given full detail (default: 800).",
    )
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=3,
        help="Minimum fashion mentions for a character to be included (default: 3).",
    )
    parser.add_argument(
        "--top-clothes",
        type=int,
        default=12,
        help="Top clothing items to keep per character (default: 12).",
    )
    parser.add_argument(
        "--top-adjs",
        type=int,
        default=15,
        help="Top fashion adjectives to keep per character (default: 15).",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=6,
        help="Maximum context sentences per (character, item) (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )
    print(f"Loading {input_path}")
    df = pd.read_parquet(input_path).reset_index()
    df = df[df["decade"].isin(DECADES)]
    print(f"{len(df):,} mentions across {df['book_id'].nunique():,} books")

    stats = compute_character_stats(df)
    print(f"{len(stats):,} characters")

    rankings, scatter_ids, detail_pairs = select_pool(
        stats, args.top_cards, args.scatter_n, args.min_mentions
    )
    print(f"Building detail for {len(detail_pairs):,} characters")

    metadata = load_metadata()
    characters = build_character_detail(
        df,
        stats,
        detail_pairs,
        metadata,
        args.top_clothes,
        args.top_adjs,
        args.max_sentences,
    )

    output = {
        "config": {
            "top_cards": args.top_cards,
            "scatter_n": args.scatter_n,
            "min_mentions": args.min_mentions,
        },
        "characters": characters,
        "rankings": rankings,
        "scatter": scatter_ids,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "unique_items.json"
    with open(out_path, "w", encoding="utf-8") as file:
        # allow_nan=False so a stray NaN raises here instead of silently
        # producing invalid JSON that the browser refuses to parse.
        json.dump(output, file, ensure_ascii=False, allow_nan=False)

    print(f"Saved {len(characters):,} characters, {len(scatter_ids):,} scatter points")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
