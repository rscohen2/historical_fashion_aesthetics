"""Log-odds of SAE codes for fashion terms.

Monroe et al. (2008) Dirichlet-prior log-odds, as in logodds.py, but the "words"
are the SAE codes active on each mention's sentence instead of character
adjectives. Documents are lowercased fashion `term`s, and each mention (duplicates
across characters dropped) counts once per active code.

Each code is shown with its LLM label from sae_labels.py, if that has been run.

Requires embeddings.py, sae.py and sae_inference.py to have been run first.

Output: data/analysis/sae_logodds/
    sae_logodds.parquet     one row per (term, code)
    sae_logodds.json        per term, its top codes, plus code labels, for the
                            visualize/ SAE Log-Odds page
    examples/{i}.json       example sentences (with book ids) per code for
                            term i of the JSON, loaded lazily by the page
    trends.json             per-decade activations of each of those codes, split
                            by its most associated terms, for the SAE Code
                            Trends page
    term_trends/{i}.json    per-decade activations of term i's top codes, loaded
                            lazily by that page's by-item view
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

from fashion.analysis.embeddings import OUTPUT_DIR as EMBEDDINGS_PATH
from fashion.analysis.sae import OUTPUT_DIR as SAE_PATH
from fashion.analysis.sae_labels import OUTPUT_DIR as LABELS_PATH
from fashion.analysis.utils import monroe_logodds
from fashion.paths import DATA_DIR

EXPERIMENT = "sae_logodds"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT


def load_mentions(input_path: Path) -> pd.DataFrame:
    """One row per mention: its lowercased `term`, `sentence_idx` into the codes,
    `decade` (e.g. 1880; NaN if the book is undated), and `book_id`."""
    df = pd.read_parquet(input_path, columns=["mention_id", "term", "decade"])
    sentence_idx = pd.read_parquet(EMBEDDINGS_PATH / "sentence_idx.parquet")
    assert len(df) == len(sentence_idx), "embeddings.py was run on a different input"
    df = df.reset_index().assign(sentence_idx=sentence_idx.sentence_idx.to_numpy())
    df = df.drop_duplicates(subset=["book_id", "mention_id"])
    # `decade` is stored in tens of years (188.0 for the 1880s).
    df = df.assign(term=df.term.str.lower(), decade=df.decade * 10)
    return df[["term", "sentence_idx", "decade", "book_id"]]


def count_matrix(
    mentions: pd.DataFrame, indices: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """(terms × codes) matrix: how many of each term's mentions activate each code."""
    term_ids, terms = pd.factorize(mentions.term)
    n_codes = int(indices.max()) + 1
    s = mentions.sentence_idx.to_numpy()
    cells = (term_ids[:, None] * n_codes + indices[s])[values[s] > 0]
    mat = np.bincount(cells, minlength=len(terms) * n_codes)
    return mat.reshape(len(terms), n_codes).astype(np.float64), list(terms)


def top_sentences(
    s: np.ndarray, indices: np.ndarray, values: np.ndarray, code: int, n: int
) -> np.ndarray:
    """Of sentence ids `s`, the (up to) `n` on which `code` fires most strongly.

    `indices` and `values` are the codes of those sentences, i.e. already
    indexed by `s`, so callers can reuse them across codes.
    """
    act = np.where(indices == code, values, 0).max(axis=1)
    order = np.argsort(-act)[:n]
    return s[order[act[order] > 0]]


def trend_counts(
    mentions: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
    pairs: pd.DataFrame,
    codes: np.ndarray,
    decades: list[int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-decade counts for the trends page, over the dated mentions.

    Returns the activations of each (term, code) row of `pairs` (pairs ×
    decades), the activations of each of `codes` over all terms (codes ×
    decades), and each term's mentions (terms × decades, indexed by term).
    """
    m = mentions[mentions.decade.isin(decades)]
    n_dec, n_codes = len(decades), int(indices.max()) + 1
    dec_ids = pd.Index(decades).get_indexer(m.decade.astype(int))
    term_ids, terms = pd.factorize(m.term)
    s = m.sentence_idx.to_numpy()
    codes_k, active = indices[s], values[s] > 0

    term_mentions = np.bincount(
        term_ids * n_dec + dec_ids, minlength=len(terms) * n_dec
    ).reshape(len(terms), n_dec)

    dec_k = np.broadcast_to(dec_ids[:, None], codes_k.shape)
    code_totals = np.bincount(
        (codes_k * n_dec + dec_k)[active], minlength=n_codes * n_dec
    ).reshape(n_codes, n_dec)[codes]

    # Map each wanted (term, code) cell to its row in `pairs`, -1 elsewhere.
    lookup = np.full(len(terms) * n_codes, -1, dtype=np.int32)
    pair_terms = terms.get_indexer(pairs.term)
    found = pair_terms >= 0  # terms with no dated mentions stay all-zero
    lookup[pair_terms[found] * n_codes + pairs.code.to_numpy()[found]] = np.flatnonzero(
        found
    )
    pair_ids = lookup[term_ids[:, None] * n_codes + codes_k]
    keep = active & (pair_ids >= 0)
    pair_counts = np.bincount(
        pair_ids[keep] * n_dec + dec_k[keep], minlength=len(pairs) * n_dec
    ).reshape(len(pairs), n_dec)

    term_frame = pd.DataFrame(term_mentions, index=terms, columns=decades)
    return pair_counts, code_totals, term_frame


def write_trends(
    results: pd.DataFrame,
    top: pd.DataFrame,
    terms: list[str],
    mentions: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
    codes: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Write the per-decade data for the SAE Code Trends page.

    trends.json has, for every code of `codes`, its `--trend-terms` most
    associated terms (z > 0); the page folds the rest into "other".
    term_trends/{i}.json has, for term i of `terms`, the codes of its rows in
    `top` (the codes sae_logodds.json lists for it).
    """
    dated = mentions.dropna(subset=["decade"])
    decade_mentions = dated.decade.astype(int).value_counts().sort_index()
    decades = decade_mentions.index[
        decade_mentions >= args.min_decade_mentions
    ].tolist()
    pairs = (
        results[(results.score > 0) & results.code.isin(codes)]
        .sort_values("score", ascending=False)
        .groupby("code")
        .head(args.trend_terms)
        .reset_index(drop=True)
    )
    # Count both views' pairs in one pass; most of them overlap.
    all_pairs = pd.concat([pairs[["term", "code"]], top[["term", "code"]]])
    all_pairs = all_pairs.drop_duplicates().reset_index(drop=True)
    all_counts, code_totals, term_mentions = trend_counts(
        mentions, indices, values, all_pairs, codes, decades
    )
    row_of = {key: i for i, key in enumerate(zip(all_pairs.term, all_pairs.code))}
    pair_counts = all_counts[[row_of[key] for key in zip(pairs.term, pairs.code)]]

    items_by_code = {}
    for row, counts in zip(pairs.itertuples(), pair_counts):
        items_by_code.setdefault(int(row.code), []).append(
            {
                "term": row.term,
                "score": round(float(row.score), 3),
                "count": int(row.count),
                "n": counts.tolist(),
            }
        )
    payload = {
        "decades": decades,
        "mentions": decade_mentions[decades].astype(int).tolist(),
        "terms": {
            term: term_mentions.loc[term].astype(int).tolist()
            for term in pairs.term.unique()
            if term in term_mentions.index
        },
        "codes": {
            str(code): {
                "total": totals.tolist(),
                "items": items_by_code.get(int(code), []),
            }
            for code, totals in zip(codes, code_totals)
        },
        "setup": {
            "trend_terms": args.trend_terms,
            "min_decade_mentions": args.min_decade_mentions,
        },
    }
    write_json(OUTPUT_DIR / "trends.json", payload)
    print(
        f"  → {OUTPUT_DIR / 'trends.json'} ({len(codes)} codes, {len(decades)} decades)"
    )

    term_dir = OUTPUT_DIR / "term_trends"
    if term_dir.exists():
        shutil.rmtree(term_dir)
    term_dir.mkdir(parents=True)
    zeros = [0] * len(decades)
    codes_by_term = top.groupby("term").code.apply(list)
    for i, term in enumerate(terms):
        has_dated = term in term_mentions.index
        term_codes = codes_by_term.get(term, [])
        write_json(
            term_dir / f"{i}.json",
            {
                "mentions": (
                    term_mentions.loc[term].astype(int).tolist() if has_dated else zeros
                ),
                "codes": {
                    str(code): all_counts[row_of[(term, code)]].tolist()
                    for code in term_codes
                },
            },
        )
    print(f"  → {term_dir}/ ({len(terms)} terms)")


def load_labels(path: Path) -> pd.DataFrame:
    """Code labels from sae_labels.py, indexed by code (empty if not run yet)."""
    if not path.exists():
        print(f"No labels at {path}; run sae_labels.py to label the codes.")
        return pd.DataFrame(columns=["label", "description"])
    labels = pd.read_parquet(path, columns=["code", "label", "description"])
    return labels.dropna(subset=["label"]).set_index("code")


def clean(sentence: str) -> str:
    return " ".join(sentence.split())


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        # allow_nan=False so a stray NaN fails here rather than in the browser.
        json.dump(
            payload, file, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )


def write_viz(
    results: pd.DataFrame,
    labels: pd.DataFrame,
    mentions: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
    sentences: pd.Series,
    args: argparse.Namespace,
) -> None:
    """Write the JSON index and per-term example files for the visualize/ page.

    Every term in the matrix is kept, most-mentioned first, with its
    `--json-codes` most associated codes (z > 0). Each example is
    `{s: sentence, bk: book_id}`; a sentence found in several books is credited
    to the first.
    """
    term_mentions = mentions.term.value_counts()
    term_sentences = mentions.groupby("term").sentence_idx.unique()
    sentence_book = (
        mentions.drop_duplicates("sentence_idx").set_index("sentence_idx").book_id
    )
    terms = term_mentions.index[term_mentions.index.isin(results.term.unique())]
    top = (
        results[results.score > 0]
        .sort_values("score", ascending=False)
        .groupby("term")
        .head(args.json_codes)
    )
    top_by_term = dict(list(top.groupby("term")))

    examples_dir = OUTPUT_DIR / "examples"
    if examples_dir.exists():
        shutil.rmtree(examples_dir)
    examples_dir.mkdir(parents=True)

    entries = []
    for i, term in enumerate(terms):
        rows = top_by_term.get(term, top.iloc[:0])
        s = term_sentences[term]
        indices_s, values_s = indices[s], values[s]
        examples = {
            int(code): [
                {"s": clean(sentences[j]), "bk": sentence_book[j]}
                for j in top_sentences(s, indices_s, values_s, code, args.examples)
            ]
            for code in rows.code
        }
        write_json(examples_dir / f"{i}.json", examples)
        entries.append(
            {
                "term": term,
                "mentions": int(term_mentions[term]),
                "codes": [
                    {
                        "code": int(row.code),
                        "score": round(float(row.score), 3),
                        "logodds": round(float(row.logodds), 3),
                        "count": int(row.count),
                    }
                    for row in rows.itertuples()
                ],
            }
        )

    shown = top.code.unique()
    payload = {
        "terms": entries,
        "labels": {
            str(code): {"label": row.label, "description": row.description}
            for code, row in labels[labels.index.isin(shown)].iterrows()
        },
        "setup": {
            "min_count": args.min_count,
            "json_codes": args.json_codes,
            "examples": args.examples,
        },
    }
    write_json(OUTPUT_DIR / "sae_logodds.json", payload)
    print(f"  → {OUTPUT_DIR / 'sae_logodds.json'} ({len(entries)} terms)")
    write_trends(
        results, top, list(terms), mentions, indices, values, np.sort(shown), args
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "terms",
        nargs="*",
        help="Fashion terms to print the top codes for (default: the --top-terms most frequent).",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--trainer-dir",
        type=Path,
        default=SAE_PATH / "trainer_0",
        help="SAE directory holding codes.pt (default: data/analysis/sae/trainer_0).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH / "labels.parquet",
        help="Code labels from sae_labels.py (default: data/analysis/sae_labels/labels.parquet).",
    )
    parser.add_argument("--top-terms", type=int, default=20)
    parser.add_argument(
        "--top-codes", type=int, default=10, help="Codes to print per term."
    )
    parser.add_argument(
        "--json-codes",
        type=int,
        default=30,
        help="Codes per term to write to the visualization JSON (default: 30).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Example sentences per (term, code), printed and in the JSON (default: 5).",
    )
    parser.add_argument(
        "--trend-terms",
        type=int,
        default=20,
        help="Terms per code with their own series in trends.json (default: 20).",
    )
    parser.add_argument(
        "--min-decade-mentions",
        type=int,
        default=1000,
        help="Leave decades with fewer dated mentions out of trends.json (default: 1000).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Drop terms and codes with fewer than N (mention, code) pairs (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = DATA_DIR / (
        "sample_for_analysis.parquet" if args.debug else "final_for_analysis.parquet"
    )

    codes = t.load(args.trainer_dir / "codes.pt")
    indices = codes["indices"].numpy()
    values = codes["values"].float().numpy()
    mentions = load_mentions(input_path)

    mat, terms = count_matrix(mentions, indices, values)
    # Rare terms and codes are left out of the prior too, as in logodds.py.
    keep_terms = mat.sum(axis=1) >= args.min_count
    keep_codes = np.flatnonzero(mat.sum(axis=0) >= args.min_count)
    mat = mat[keep_terms][:, keep_codes]
    terms = [term for term, keep in zip(terms, keep_terms) if keep]
    print(f"Matrix: {len(terms)} terms × {len(keep_codes)} codes")

    delta, sigma = monroe_logodds(mat)
    labels = load_labels(args.labels)
    results = pd.DataFrame(
        {
            "term": np.repeat(terms, len(keep_codes)),
            "code": np.tile(keep_codes, len(terms)),
            "logodds": delta.ravel(),
            "sigma": sigma.ravel(),
            "score": (delta / sigma).ravel(),
            "count": mat.ravel().astype(int),
        }
    ).join(labels, on="code")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sae_logodds.parquet"
    results.to_parquet(out_path, index=False)
    print(f"  → {out_path}")

    sentences = pd.read_parquet(EMBEDDINGS_PATH / "sentences.parquet").sentence
    write_viz(results, labels, mentions, indices, values, sentences, args)

    term_sentences = mentions.groupby("term").sentence_idx.unique()
    show = args.terms or mentions.term.value_counts().index[: args.top_terms]
    for term in show:
        top = results[results.term.eq(term)].nlargest(args.top_codes, "score")
        if top.empty:
            print(f"\n=== {term}: not in the matrix ===")
            continue
        print(f"\n=== {term} ===")
        s = term_sentences[term]
        indices_s, values_s = indices[s], values[s]
        for row in top.itertuples():
            label = "" if pd.isna(row.label) else f" [{row.label}]"
            print(f"code {row.code}{label}: z = {row.score:.1f} (n = {row.count})")
            for j in top_sentences(s, indices_s, values_s, row.code, args.examples):
                print(f"\t{clean(sentences[j])}")


if __name__ == "__main__":
    main()
