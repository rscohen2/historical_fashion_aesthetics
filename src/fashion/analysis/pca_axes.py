"""Principal axes of characters' fashion vocabulary, tested against gender and genre.

Rebuilds the character × fashion-term matrix from notebooks/pca.ipynb (each
named character with >= --min-count mentions, each term stem with >= --min-count
mentions, cells = mention counts), decomposes it with a truncated SVD, and tests
every component two ways:

  - gender (character level): Kendall's τ between a character's PC score and
    their gender (male = 0, female = 1), over male/female characters only.
  - genre (book level): a book's position is the mean PC score of its
    characters; for each genre, the mean pairwise distance between its books is
    compared with 10,000 random relabelings. A ratio below 1 means the genre's
    books sit closer together on that PC than chance. The omnibus p tests the
    mean of those distances across genres.

--normalize picks what each character's row holds:
  - counts: raw mention counts (as in the notebook). Heavily described
            characters dominate, and a component can end up tracking how much
            a character is described rather than how.
  - l2:     each row scaled to unit length first, so characters are compared
            on the mix of terms they wear, not on how many mentions they have.

--centering picks how the matrix is centered before the SVD (after
normalizing):
  - term: subtract each term's mean over characters (standard PCA).
  - row:  subtract each character's mean over terms, as the notebook did. The
          first component then mostly tracks common vs. rare vocabulary.

Output: data/analysis/pca_axes/{centering}[_{normalize}]/  (no suffix for counts)
          pc_summary.csv    per PC: variance explained, gender τ, genre omnibus p,
                            and Spearman ρ with a character's total mentions
          genre_tests.csv   per PC × genre: n, observed, null mean, ratio, p
          loadings.csv      per term stem: loading on each PC, mention count
          characters.parquet  per character: book, gender, PC scores
          books.parquet       per book: genres, number of characters, PC scores
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import TruncatedSVD

from fashion.paths import DATA_DIR

EXPERIMENT = "pca_axes"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
HATHI_META = DATA_DIR / "hathimeta" / "titlemeta.tsv"
GENRE_MERGES = DATA_DIR / "hathi_genre_merges.csv"


def build_matrix(df: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Character × term-stem mention counts, plus each character's book and gender.

    Unnamed characters (character_id 0) are dropped. Terms and characters with
    fewer than `min_count` mentions are dropped together, from the unfiltered
    counts, as in the notebook.
    """
    df = df[df.character_id.ne(0)].copy()
    df["char"] = (df.book_id + "-" + df.character_id.astype(str)).str.lower()
    df["stem"] = df.term_stem.str.lower()
    term_counts = df.stem.value_counts()
    char_counts = df.char.value_counts()
    df = df[
        df.stem.isin(term_counts[term_counts >= min_count].index)
        & df.char.isin(char_counts[char_counts >= min_count].index)
    ]
    counts = pd.crosstab(df.char, df.stem)
    chars = (
        df.drop_duplicates("char")
        .set_index("char")[["book_id", "gender_dep_pron"]]
        .rename(columns={"gender_dep_pron": "gender"})
        .loc[counts.index]
    )
    return counts, chars


def decompose(
    counts: pd.DataFrame, normalize: str, centering: str, n_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores (chars × PCs), loadings (PCs × terms), and variance explained.

    Each component's sign is fixed so its largest-magnitude loading is
    positive; SVD signs are otherwise arbitrary and would flip between runs.
    """
    x = counts.to_numpy(dtype=np.float64)
    if normalize == "l2":
        # Every character passed the mention floor, so no row is all zeros.
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    axis = 0 if centering == "term" else 1
    x = x - x.mean(axis=axis, keepdims=True)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    scores = svd.fit_transform(x)
    loadings = svd.components_
    signs = np.sign(loadings[np.arange(len(loadings)), np.abs(loadings).argmax(axis=1)])
    return scores * signs, loadings * signs[:, None], svd.explained_variance_ratio_


def gender_tau(scores: np.ndarray, gender: pd.Series) -> pd.DataFrame:
    """Kendall's τ of each PC with gender (male = 0, female = 1)."""
    mask = gender.isin(["male", "female"]).to_numpy()
    female = gender[mask].eq("female").astype(int).to_numpy()
    rows = []
    for pc in range(scores.shape[1]):
        result = kendalltau(scores[mask, pc], female)
        rows.append({"gender_tau": result.statistic, "gender_p": result.pvalue})
    return pd.DataFrame(rows)


def load_genres(book_ids: set[str]) -> pd.DataFrame:
    """(book_id, genre) pairs for fiction genres, after the notebook's genre merges.

    Each pair appears once, even if several raw genres merge into the same one.
    """
    meta = pd.read_csv(HATHI_META, sep="\t", usecols=["docid", "genres"], dtype=str)
    meta["book_id"] = meta.docid + ".clean"
    meta = meta[meta.book_id.isin(book_ids)].dropna(subset=["genres"])
    genre_map = pd.read_csv(
        GENRE_MERGES, names=["original", "clean"], index_col=0
    ).clean.to_dict()
    pairs = meta[["book_id", "genres"]].assign(genre=meta.genres.str.split("|"))
    pairs = pairs.explode("genre")
    pairs["genre"] = pairs.genre.map(genre_map)
    pairs = pairs.dropna(subset=["genre"])
    pairs = pairs[pairs.genre.str.contains("fiction", case=False)]
    return pairs[["book_id", "genre"]].drop_duplicates().reset_index(drop=True)


def group_dispersion(x: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean |xi - xj| over all pairs within each group (NaN for groups under 2).

    Sorting within groups turns the pairwise sum into a weighted sum: the k-th
    smallest of n values appears with weight (2k - n + 1).
    """
    order = np.lexsort((x, codes))
    xs, cs = x[order], codes[order]
    sizes = np.bincount(cs, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    rank = np.arange(len(xs)) - starts[cs]
    weighted = np.bincount(
        cs, weights=(2 * rank - sizes[cs] + 1) * xs, minlength=n_groups
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sizes >= 2, weighted / (sizes * (sizes - 1) / 2), np.nan)


def genre_permutation_test(
    x: np.ndarray, labels: np.ndarray, n_perm: int, seed: int
) -> tuple[pd.DataFrame, float]:
    """Per-genre and omnibus one-sided tests that genres are tighter than chance."""
    genres, codes, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    observed = group_dispersion(x, codes, len(genres))
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(genres)))
    for i in range(n_perm):
        null[i] = group_dispersion(x, rng.permutation(codes), len(genres))

    p = ((null <= observed).sum(axis=0) + 1) / (n_perm + 1)
    p[np.isnan(observed)] = np.nan
    p_all = ((np.nanmean(null, axis=1) <= np.nanmean(observed)).sum() + 1) / (
        n_perm + 1
    )
    table = pd.DataFrame(
        {
            "genre": genres,
            "n": sizes,
            "observed": observed,
            "null_mean": null.mean(axis=0),
            "p_value": p,
        }
    )
    table["ratio"] = table.observed / table.null_mean
    return table, float(p_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--normalize",
        choices=["counts", "l2"],
        default="counts",
        help="Raw mention counts (the notebook) or unit-length character rows.",
    )
    parser.add_argument(
        "--centering",
        choices=["term", "row", "both"],
        default="both",
        help="Center by term (standard PCA), by character (the notebook), or run both.",
    )
    parser.add_argument(
        "--n-components", type=int, default=20, help="Components to keep (default: 20)."
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum mentions for a character or term stem (default: 10).",
    )
    parser.add_argument(
        "--min-genre-books",
        type=int,
        default=1,
        help=(
            "Genres with fewer books are left out of the genre test (default: 1, "
            "as in the notebook; single-book genres join the permutation pool but "
            "get no p-value)."
        ),
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10_000,
        help="Permutations per PC for the genre test (default: 10,000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="SVD seed (default: 42).")
    parser.add_argument(
        "--perm-seed", type=int, default=0, help="Permutation seed (default: 0)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Root output directory."
    )
    return parser.parse_args()


def run_name(centering: str, normalize: str) -> str:
    """Output folder: the centering, suffixed by the normalization unless raw."""
    return centering if normalize == "counts" else f"{centering}_{normalize}"


def run(
    counts: pd.DataFrame,
    chars: pd.DataFrame,
    genres: pd.DataFrame,
    centering: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    scores, loadings, variance = decompose(
        counts, args.normalize, centering, args.n_components, args.seed
    )
    pcs = [f"PC{i + 1}" for i in range(scores.shape[1])]

    summary = pd.DataFrame({"pc": pcs, "variance_explained": variance})
    summary = pd.concat([summary, gender_tau(scores, chars.gender)], axis=1)
    # A PC that mostly tracks how much a character is described correlates
    # with their total mentions.
    total = counts.sum(axis=1).to_numpy()
    summary["mentions_rho"] = [
        spearmanr(scores[:, i], total).statistic for i in range(len(pcs))
    ]

    char_scores = pd.concat(
        [chars.reset_index(), pd.DataFrame(scores, columns=pcs)], axis=1
    )
    book_scores = char_scores.groupby("book_id")[pcs].mean()
    book_scores.insert(0, "n_characters", char_scores.groupby("book_id").size())

    tested = genres[genres.book_id.isin(book_scores.index)]
    sizes = tested.genre.value_counts()
    tested = tested[tested.genre.isin(sizes[sizes >= args.min_genre_books].index)]
    tables, omnibus = [], []
    for pc in pcs:
        x = book_scores.loc[tested.book_id, pc].to_numpy()
        table, p_all = genre_permutation_test(
            x, tested.genre.to_numpy(), args.n_perm, args.perm_seed
        )
        tables.append(table.assign(pc=pc))
        omnibus.append(p_all)
    genre_tests = pd.concat(tables, ignore_index=True)
    summary["genre_p"] = omnibus
    summary["genre_mean_ratio"] = (
        genre_tests.groupby("pc", sort=False).ratio.mean().to_numpy()
    )

    out = args.output_dir / run_name(centering, args.normalize)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "pc_summary.csv", index=False)
    genre_tests[
        ["pc", "genre", "n", "observed", "null_mean", "ratio", "p_value"]
    ].to_csv(out / "genre_tests.csv", index=False)
    loading_table = pd.DataFrame(loadings.T, index=counts.columns, columns=pcs)
    loading_table.insert(0, "mentions", counts.sum(axis=0))
    loading_table.rename_axis("term").to_csv(out / "loadings.csv")
    char_scores.to_parquet(out / "characters.parquet")
    genre_lists = tested.groupby("book_id").genre.agg(list)
    book_scores.insert(0, "genres", genre_lists.reindex(book_scores.index))
    book_scores.to_parquet(out / "books.parquet")
    print(f"Saved results to {out}")
    return summary


def describe_ends(loadings_path: Path, pc: str, n: int = 6, min_mentions: int = 1000):
    table = pd.read_csv(loadings_path, index_col="term")
    common = table[table.mentions > min_mentions][pc].sort_values()
    return list(common.index[:n]), list(common.index[::-1][:n])


def main() -> None:
    args = parse_args()
    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )
    df = pd.read_parquet(
        input_path, columns=["term_stem", "gender_dep_pron"]
    ).reset_index()
    counts, chars = build_matrix(df, args.min_count)
    print(f"Matrix: {counts.shape[0]:,} characters × {counts.shape[1]} term stems")

    genres = load_genres(set(chars.book_id))
    print(f"{genres.book_id.nunique():,} books with a fiction genre")

    centerings = ["term", "row"] if args.centering == "both" else [args.centering]
    for centering in centerings:
        summary = run(counts, chars, genres, centering, args)
        print(f"\n{run_name(centering, args.normalize)}, first 10 PCs:")
        print(
            summary.head(10).to_string(index=False, float_format=lambda v: f"{v:.3g}")
        )
        loadings_path = (
            args.output_dir / run_name(centering, args.normalize) / "loadings.csv"
        )
        for pc in itertools.islice(summary.pc, 3):
            low, high = describe_ends(loadings_path, pc)
            print(f"  {pc}: {', '.join(low)}  <->  {', '.join(high)}")


if __name__ == "__main__":
    main()
