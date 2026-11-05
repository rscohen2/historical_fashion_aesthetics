"""Corpus-level log-odds of character adjectives for fashion terms.

Uses Monroe et al. (2008) Dirichlet-prior log-odds to measure which character
adjectives are statistically associated with each fashion term across the
whole corpus (no temporal split). Character adjectives are the "words" and
fashion terms are the "documents".

Reproduces the two setups in notebooks/fashion_gender_classifier.ipynb:
  - default (the corpus-wide cell): one count per mention per character
    adjective, documents are lowercased `term`s.
  - --gender {male,female} (the "Gender-controlled log-odds" cell): only that
    gender's characters, one count per (book, character, term) using the first
    mention's adjectives, documents are lowercased `term_stem`s.
Both drop terms and adjectives seen fewer than --min-count times before the
matrix is built, so they are excluded from the prior as well.

Output: data/analysis/logodds/logodds.json (default)
        data/analysis/logodds/logodds_{gender}.json (--gender)
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from fashion.analysis.utils import monroe_logodds
from fashion.paths import DATA_DIR

EXPERIMENT = "logodds"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
DOC = "doc"
WORD = "word"


def prepare_units(df: pd.DataFrame, gender: str | None) -> pd.DataFrame:
    """One row per counting unit, with its document label in `DOC`.

    Corpus-wide, a unit is a mention (duplicates of a mention across characters
    dropped). With `gender`, a unit is a (book, character, term) triple
    reduced to its first mention, as in the notebook's gender-controlled cell.
    """
    df = df.drop_duplicates(subset=["book_id", "mention_id"])
    if gender is None:
        units = df.copy()
        units[DOC] = units["term"].str.lower()
    else:
        units = (
            df[df.gender_dep_pron.eq(gender)]
            .groupby(["book_id", "character_id", "term"])
            .agg("first")
            .reset_index()
        )
        units[DOC] = units["term_stem"].str.lower()
    return units


def explode_pairs(units: pd.DataFrame, col_word: str, min_count: int) -> pd.DataFrame:
    """(doc, word) pairs, one per unit per adjective, with rare docs/words dropped.

    Terms and adjectives seen fewer than `min_count` times are removed before
    the matrix is built, so they are left out of the prior too.
    """
    pairs = units[[DOC, col_word]].explode(col_word).dropna()
    pairs = pairs.rename(columns={col_word: WORD})
    pairs[WORD] = pairs[WORD].astype(str).str.lower()
    doc_counts = pairs[DOC].value_counts()
    word_counts = pairs[WORD].value_counts()
    return pairs[
        pairs[DOC].isin(doc_counts[doc_counts >= min_count].index)
        & pairs[WORD].isin(word_counts[word_counts >= min_count].index)
    ]


def build_count_matrix(
    pairs: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, int], list[str], dict[str, int]]:
    """Doc-term count matrix: how many (doc, word) pairs share each cell."""
    counts = pd.crosstab(pairs[DOC], pairs[WORD])
    docs = [str(d) for d in counts.index]
    words = [str(w) for w in counts.columns]
    doc2idx = {d: i for i, d in enumerate(docs)}
    word2idx = {w: i for i, w in enumerate(words)}
    return counts.to_numpy(dtype=np.float64), docs, doc2idx, words, word2idx


def compute_logodds(
    mat: np.ndarray,
    docs: list[str],
    doc2idx: dict[str, int],
    words: list[str],
    term: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Monroe et al. (2008) Dirichlet-prior log-odds for one term vs. all others."""
    delta, sigma = monroe_logodds(mat)
    return delta[doc2idx[term]], sigma[doc2idx[term]]


def select_top_logodds(
    mat: np.ndarray,
    docs: list[str],
    doc2idx: dict[str, int],
    words: list[str],
    top_terms: list[str],
    top_adj: int,
) -> list[dict]:
    """Compute log-odds for each top term; return top adjectives per term by |score|."""
    rows: list[dict] = []
    for term in tqdm(top_terms, desc="Computing log-odds"):
        delta, sigma = compute_logodds(mat, docs, doc2idx, words, term)
        score = delta / sigma
        term_idx = doc2idx[term]
        candidates = []
        for j, word in enumerate(words):
            count = int(mat[term_idx][j])
            candidates.append(
                {
                    "term": term,
                    "adjective": word,
                    "logodds": float(delta[j]),
                    "sigma": float(sigma[j]),
                    "score": float(score[j]),
                    "count": count,
                }
            )
        candidates.sort(key=lambda r: abs(r["score"]), reverse=True)
        rows.extend(candidates[:top_adj])
    return rows


def collect_examples(
    df: pd.DataFrame,
    term_adj_pairs: list[tuple[str, str]],
    col_doc: str,
    col_word: str,
    max_per_pair: int = 10,
) -> dict[str, dict[str, list[dict]]]:
    """Collect example sentences for each (term, adj) pair with early stopping.

    Each example is ``{s: sentence, bk: book_id}``; the page looks the book up
    in ``books.py``'s output.
    """
    by_term: dict[str, set[str]] = defaultdict(set)
    for term, adj in term_adj_pairs:
        by_term[term].add(adj)

    examples: dict[str, dict[str, list[dict]]] = {
        term: {adj: [] for adj in adjs} for term, adjs in by_term.items()
    }

    def _clean(s: str) -> str:
        s = s.replace("\n", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
        return re.sub(r"\s+", " ", s).strip()

    for term, target_adjs in tqdm(by_term.items(), desc="Collecting examples"):
        still_need = set(target_adjs)
        term_df = df[df[col_doc] == term]
        for row in term_df.itertuples(index=False):
            if not still_need:
                break
            adjs = getattr(row, col_word)
            if not isinstance(adjs, (list, np.ndarray)):
                continue
            sentence = _clean(row.sentence)
            done: set[str] = set()
            # A set, so an adjective repeated within a mention adds it only once.
            for adj_lower in {str(adj).lower() for adj in adjs if adj is not None}:
                if adj_lower in still_need:
                    bucket = examples[term][adj_lower]
                    if len(bucket) < max_per_pair:
                        bucket.append({"s": sentence, "bk": row.book_id})
                        if len(bucket) >= max_per_pair:
                            done.add(adj_lower)
            still_need -= done

    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default=None,
        help=(
            "Gender-controlled setup: only this gender's characters, one count per "
            "(book, character, term), documents are term stems (default: corpus-wide)."
        ),
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=20,
        help="Number of top fashion terms to analyze (default: 20).",
    )
    parser.add_argument(
        "--top-adj",
        type=int,
        default=30,
        help="Number of top character adjectives to keep per term by |score| (default: 30).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help=(
            "Drop terms and adjectives with fewer than N (unit, adjective) pairs "
            "before building the matrix (default: 10)."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum example sentences per (term, adj) pair (default: 10).",
    )
    parser.add_argument(
        "--col-word",
        type=str,
        default="adjectives_char",
        help="Column to use as word / character adjective (default: adjectives_char).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df = pd.read_parquet(input_path).reset_index()
    units = prepare_units(df, args.gender)
    pairs = explode_pairs(units, args.col_word, args.min_count)

    mat, docs, doc2idx, words, _word2idx = build_count_matrix(pairs)
    print(f"Matrix: {len(docs)} terms × {len(words)} adjectives")

    # Most frequent terms by unit count, among those that survived the filter.
    term_counts = units[DOC].value_counts()
    top_terms = [t for t in term_counts.index if t in doc2idx][: args.top_terms]
    print(f"Top {args.top_terms} terms: {top_terms}")

    logodds_rows = select_top_logodds(
        mat, docs, doc2idx, words, top_terms, args.top_adj
    )

    # Examples come from every mention of the term, not just the counted units.
    sentences = df.drop_duplicates(subset=["book_id", "mention_id"])
    if args.gender is not None:
        sentences = sentences[sentences.gender_dep_pron.eq(args.gender)]
    sentences = sentences.assign(
        **{DOC: sentences["term" if args.gender is None else "term_stem"].str.lower()}
    )
    term_adj_pairs = [(row["term"], row["adjective"]) for row in logodds_rows]
    examples = collect_examples(
        sentences, term_adj_pairs, DOC, args.col_word, args.max_examples
    )

    output = {
        "terms": top_terms,
        "logodds": logodds_rows,
        "setup": {
            "gender": args.gender,
            "unit": "mention" if args.gender is None else "character",
            "min_count": args.min_count,
        },
        "examples": examples,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "logodds.json" if args.gender is None else f"logodds_{args.gender}.json"
    out_path = OUTPUT_DIR / name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    n_pairs = len(logodds_rows)
    print(f"Saved {n_pairs} (term, adj) pairs across {len(top_terms)} terms")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
