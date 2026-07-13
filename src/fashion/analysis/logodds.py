"""Corpus-level log-odds of character adjectives for fashion terms.

Uses Monroe et al. (2008) Dirichlet-prior log-odds to measure which character
adjectives are statistically associated with each fashion term across the
whole corpus (no temporal split). Character adjectives are the "words" and
fashion terms are the "documents".

Output: data/analysis/logodds/logodds.json
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from fashion.paths import DATA_DIR

EXPERIMENT = "logodds"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
DECADES = list(range(180, 192))


def build_count_matrix(
    df: pd.DataFrame,
    col_doc: str,
    col_word: str,
) -> tuple[np.ndarray, list[str], dict[str, int], list[str], dict[str, int]]:
    """Build doc-term count matrix from unique (book_id, character_id) pairs.

    Each (book_id, character_id, term) triplet contributes at most once per
    character adjective, preventing the same character from inflating counts.
    """
    pair_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for (_book, _char, term), group in tqdm(
        df.groupby(["book_id", "character_id", col_doc]),
        desc="Building count matrix",
    ):
        all_adjs: set[str] = set()
        for adjs in group[col_word]:
            if isinstance(adjs, (list, np.ndarray)):
                for adj in adjs:
                    if adj is not None:
                        all_adjs.add(str(adj).lower())
        for adj in all_adjs:
            pair_counts[str(term)][adj] += 1

    docs = sorted(pair_counts.keys())
    words: list[str] = sorted({w for adj_map in pair_counts.values() for w in adj_map})

    doc2idx = {d: i for i, d in enumerate(docs)}
    word2idx = {w: i for i, w in enumerate(words)}

    mat = np.zeros((len(docs), len(words)), dtype=np.float64)
    for term, adj_counts in pair_counts.items():
        for adj, count in adj_counts.items():
            mat[doc2idx[term]][word2idx[adj]] = count

    return mat, docs, doc2idx, words, word2idx


def compute_logodds(
    mat: np.ndarray,
    docs: list[str],
    doc2idx: dict[str, int],
    words: list[str],
    term: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Monroe et al. (2008) Dirichlet-prior log-odds for one term vs. all others."""
    term_idx = doc2idx[term]
    other_idx = [i for i in range(len(docs)) if i != term_idx]

    g1 = mat[term_idx]
    g2 = mat[other_idx].sum(axis=0)
    n1 = float(g1.sum())
    n2 = float(g2.sum())
    prior = mat.sum(axis=0)
    n_prior = float(prior.sum())

    odds1 = (g1 + prior) / ((n1 + n_prior) - (g1 + prior))
    odds2 = (g2 + prior) / ((n2 + n_prior) - (g2 + prior))
    delta = np.log(odds1) - np.log(odds2)
    sigma = np.sqrt(1.0 / (g1 + prior) + 1.0 / (g2 + prior))
    return delta, sigma


def select_top_logodds(
    mat: np.ndarray,
    docs: list[str],
    doc2idx: dict[str, int],
    words: list[str],
    top_terms: list[str],
    min_count: int,
    top_adj: int,
) -> list[dict]:
    """Compute log-odds for each top term; return top adjectives per term by |score|."""
    rows: list[dict] = []
    for term in tqdm(top_terms, desc="Computing log-odds"):
        delta, sigma = compute_logodds(mat, docs, doc2idx, words, term)
        score = delta / sigma
        term_idx = doc2idx[term]
        # Collect all adj rows meeting min_count, sorted by |score| descending
        candidates = []
        for j, word in enumerate(words):
            count = int(mat[term_idx][j])
            if count < min_count:
                continue
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
) -> dict[str, dict[str, list[str]]]:
    """Collect example sentences for each (term, adj) pair with early stopping."""
    by_term: dict[str, set[str]] = defaultdict(set)
    for term, adj in term_adj_pairs:
        by_term[term].add(adj)

    examples: dict[str, dict[str, list[str]]] = {
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
            for adj in adjs:
                if adj is None:
                    continue
                adj_lower = str(adj).lower()
                if adj_lower in still_need:
                    bucket = examples[term][adj_lower]
                    if len(bucket) < max_per_pair:
                        bucket.append(sentence)
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
        help="Minimum character-pair count for an adjective to be included (default: 10).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum example sentences per (term, adj) pair (default: 10).",
    )
    parser.add_argument(
        "--col-doc",
        type=str,
        default="term",
        help="Column to use as document / fashion term (default: term).",
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
    df = df[df.decade.isin(DECADES)]

    term_counts = df[args.col_doc].value_counts()
    top_terms = term_counts.head(args.top_terms).index.tolist()
    print(f"Top {args.top_terms} terms: {top_terms}")

    mat, docs, doc2idx, words, word2idx = build_count_matrix(
        df, args.col_doc, args.col_word
    )
    print(f"Matrix: {len(docs)} terms × {len(words)} adjectives")

    logodds_rows = select_top_logodds(
        mat, docs, doc2idx, words, top_terms, args.min_count, args.top_adj
    )

    term_adj_pairs = [(row["term"], row["adjective"]) for row in logodds_rows]
    examples = collect_examples(
        df, term_adj_pairs, args.col_doc, args.col_word, args.max_examples
    )

    output = {
        "terms": top_terms,
        "logodds": logodds_rows,
        "examples": examples,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "logodds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    n_pairs = len(logodds_rows)
    print(f"Saved {n_pairs} (term, adj) pairs across {len(top_terms)} terms")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
