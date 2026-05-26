"""
Temporal log odds with gaussian process regression.


- For the character adjectives and fashion items, split into timesteps of a decade.
- In each timestep, calculate the log odds of the term being associated with each fashion item.
- Fit a gaussian process regression to the log odds over time, and plot the results.

Based on the log odds code in `notebooks/fashion_gender_classifier.ipynb`.
Structured in the same way as the other scripts in this `analysis/` directory.
"""

import argparse
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from tqdm import tqdm

from fashion.paths import DATA_DIR

EXPERIMENT = "gp_logodds"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
DECADES = list(range(180, 192))
DECADE_MIN = float(min(DECADES))
DECADE_MAX = float(max(DECADES))


@dataclass
class DocTermMatrix:
    mat: np.ndarray
    idx2word: list[str]
    word2idx: dict[str, int]
    idx2doc: list[str]
    doc2idx: dict[str, int]


def create_doc_term_mat(
    df: pd.DataFrame, document_col: str, word_col: str
) -> DocTermMatrix:
    idx2word = list(df[word_col].unique())
    word2idx = {v: k for k, v in enumerate(idx2word)}
    idx2doc = list(df[document_col].unique())
    doc2idx = {v: k for k, v in enumerate(idx2doc)}
    mat = np.zeros((len(doc2idx), len(idx2word)))
    for doc, word in zip(df[document_col], df[word_col]):
        mat[doc2idx[doc]][word2idx[word]] += 1
    return DocTermMatrix(mat, idx2word, word2idx, idx2doc, doc2idx)


def _logodds_for_term(
    matrix: DocTermMatrix, term: str
) -> tuple[np.ndarray, np.ndarray]:
    """Monroe et al. (2008) Dirichlet-prior log-odds: returns (delta, sigma) for all words."""
    mat = matrix.mat
    group1_idx = [i for i, d in enumerate(matrix.idx2doc) if d == term]
    group2_idx = [i for i, d in enumerate(matrix.idx2doc) if d != term]

    n_prior = mat.sum()
    g1 = mat[group1_idx].sum(axis=0)
    g2 = mat[group2_idx].sum(axis=0)
    n1 = float(g1.sum())
    n2 = float(g2.sum())
    prior = mat.sum(axis=0)

    odds1 = (g1 + prior) / ((n1 + n_prior) - (g1 + prior))
    odds2 = (g2 + prior) / ((n2 + n_prior) - (g2 + prior))
    delta = np.log(odds1) - np.log(odds2)
    sigma = np.sqrt(1.0 / (g1 + prior) + 1.0 / (g2 + prior))
    return delta, sigma


def compute_temporal_logodds(
    df: pd.DataFrame,
    top_terms: list[str],
    candidate_adjs: list[str],
    col_word: str = "adjectives_char",
) -> pd.DataFrame:
    """Compute per-decade log odds for each (fashion term, character adjective) pair."""
    rows: list[dict] = []
    for decade in tqdm(DECADES, desc="Computing temporal log odds"):
        decade_df = df[df.decade == decade]
        if len(decade_df) == 0:
            continue
        matrix = create_doc_term_mat(decade_df, "term", col_word)
        for term in top_terms:
            if term not in matrix.doc2idx:
                continue
            delta, sigma = _logodds_for_term(matrix, term)
            for adj in candidate_adjs:
                if adj not in matrix.word2idx:
                    continue
                j = matrix.word2idx[adj]
                rows.append(
                    {
                        "decade": float(decade),
                        "term": term,
                        "adjective": adj,
                        "logodds": float(delta[j]),
                        "sigma": float(sigma[j]),
                        "score": float(delta[j] / sigma[j]),
                        "count": int(matrix.mat[:, j].sum()),
                    }
                )
    return pd.DataFrame(rows)


def _normalize_decade(x: np.ndarray) -> np.ndarray:
    return (x - DECADE_MIN) / (DECADE_MAX - DECADE_MIN)


def fit_gp(
    decades: np.ndarray, logodds_vals: np.ndarray, sigmas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit GP to (decade, log-odds) and return (pred_decades, mean, std) on a dense grid.

    sigmas are the per-observation standard errors from the log-odds formula, used directly
    as heteroscedastic noise (alpha=sigmas**2) rather than estimated from the data.
    """
    X_train = _normalize_decade(decades).reshape(-1, 1)
    pred_decades = np.linspace(DECADE_MIN, DECADE_MAX, 100)
    X_pred = _normalize_decade(pred_decades).reshape(-1, 1)

    kernel = ConstantKernel(1.0, (1e-8, 10.0)) * Matern(
        length_scale=0.3, length_scale_bounds=(0.01, 2.0), nu=1.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=sigmas**2, normalize_y=True, n_restarts_optimizer=10
    )
    gp.fit(X_train, logodds_vals)
    mean, std = gp.predict(X_pred, return_std=True)
    return pred_decades, mean, std


def build_gp_df(logodds_df: pd.DataFrame, min_points: int = 3) -> pd.DataFrame:
    """Fit a GP per (term, adjective) pair; return long-form GP prediction DataFrame."""
    gp_rows: list[dict] = []
    for (term, adj), group in tqdm(
        logodds_df.groupby(["term", "adjective"]), desc="Fitting GPs"
    ):
        if len(group) < min_points:
            continue
        decades = group.decade.values.astype(float)
        lo = group.logodds.values.astype(float)
        sigmas = group.sigma.values.astype(float)
        try:
            pred_decades, mean, std = fit_gp(decades, lo, sigmas)
        except Exception:
            continue
        for d, m, s in zip(pred_decades, mean, std):
            gp_rows.append(
                {
                    "decade": float(d),
                    "term": term,
                    "adjective": adj,
                    "gp_mean": float(m),
                    "gp_std": float(s),
                }
            )
    return pd.DataFrame(gp_rows)


def select_top_adjs(logodds_df: pd.DataFrame, top_adj: int) -> dict[str, list[str]]:
    """For each term, pick top adjectives by mean absolute log-odds across decades."""
    result: dict[str, list[str]] = {}
    for term, group in logodds_df.groupby("term"):
        by_adj = group.groupby("adjective")["score"].apply(lambda x: x.abs().max())
        result[str(term)] = by_adj.nlargest(top_adj).index.tolist()
    return result


def make_term_chart(
    logodds_df: pd.DataFrame,
    gp_df: pd.DataFrame,
    term: str,
    top_adjs: list[str],
) -> alt.LayerChart:
    """Build an Altair chart (GP line + CI band + raw points) for one fashion term."""
    raw = logodds_df[
        logodds_df.term.eq(term) & logodds_df.adjective.isin(top_adjs)
    ].copy()
    gp = gp_df[gp_df.term.eq(term) & gp_df.adjective.isin(top_adjs)].copy()

    raw["year"] = raw.decade * 10
    gp["year"] = gp.decade * 10
    gp["ci_low"] = gp.gp_mean - 2.0 * gp.gp_std
    gp["ci_high"] = gp.gp_mean + 2.0 * gp.gp_std

    param_name = "sel_" + term.replace(" ", "_").replace("-", "_")
    selection = alt.selection_point(
        name=param_name, fields=["adjective"], bind="legend", on="click"
    )
    color_enc = alt.Color("adjective:N", title="Adjective")
    x_enc = alt.X("year:Q", title="Year", axis=alt.Axis(format="d", tickMinStep=10))

    band = (
        alt.Chart(gp)
        .mark_area(opacity=0.15)
        .encode(
            x=x_enc,
            y=alt.Y("ci_low:Q", title="Log-odds"),
            y2=alt.Y2("ci_high:Q"),
            color=color_enc,
            opacity=alt.condition(selection, alt.value(0.3), alt.value(0.03)),
        )
        .add_params(selection)
    )

    line = (
        alt.Chart(gp)
        .mark_line()
        .encode(
            x=x_enc,
            y=alt.Y("gp_mean:Q"),
            color=color_enc,
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1)),
        )
    )

    points = (
        alt.Chart(raw)
        .mark_point(filled=True, size=50)
        .encode(
            x=x_enc,
            y=alt.Y("logodds:Q"),
            color=color_enc,
            opacity=alt.condition(selection, alt.value(0.9), alt.value(0.1)),
            tooltip=["year:Q", "adjective:N", "logodds:Q"],
        )
    )

    return (band + line + points).properties(
        title=f"Adjective log-odds over time: '{term}'",
        width=600,
        height=300,
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
        "--top-terms",
        type=int,
        default=5,
        help="Number of top fashion terms to analyze (default: 5).",
    )
    parser.add_argument(
        "--top-adj",
        type=int,
        default=8,
        help="Number of top character adjectives to show per term (default: 8).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=10,
        help="Minimum occurrence count for terms and adjectives (default: 10).",
    )
    parser.add_argument(
        "--col-doc",
        type=str,
        default="term_stem",
        help="Column name to use as 'document' for log-odds calculation (default 'term_stem').",
    )
    parser.add_argument(
        "--col-word",
        type=str,
        default="adjectives_char",
        help="Column name to use as 'word' for log-odds calculation (default 'adjectives_char').",
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
    df.loc[:, "char_adj_set"] = df["adjectives_char"].apply(
        lambda x: set(w.lower() for w in x) if isinstance(x, list) else set()
    )

    # One row per (book_id, character_id, term_stem, decade) to avoid counting repeated mentions
    char_term = (
        df.groupby(["book_id", "character_id", args.col_doc, "decade"])
        .agg({args.col_word: "first"})
        .reset_index()
        .rename(columns={args.col_doc: "term"})
    )
    exploded = char_term.explode(args.col_word).dropna(subset=[args.col_word])
    exploded = exploded.copy()
    exploded[args.col_word] = exploded[args.col_word].str.lower()

    term_counts = exploded.term.value_counts()
    top_terms = (
        term_counts[term_counts >= args.min_df].head(args.top_terms).index.tolist()
    )
    print(f"Top {args.top_terms} terms: {top_terms}")

    adj_counts = exploded[args.col_word].value_counts()
    candidate_adjs = adj_counts[adj_counts >= args.min_df].index.tolist()
    print(f"Candidate adjectives: {len(candidate_adjs)}")

    subset = exploded[
        exploded.term.isin(top_terms) & exploded[args.col_word].isin(candidate_adjs)
    ]

    logodds_df = compute_temporal_logodds(
        subset, top_terms, candidate_adjs, args.col_word
    )
    top_adjs_per_term = select_top_adjs(logodds_df, args.top_adj)

    print("Top adjectives per term:")
    for term, adjs in top_adjs_per_term.items():
        print(f"  {term}: {adjs}")

    selected_pairs = pd.DataFrame(
        [
            {"term": term, "adjective": adj}
            for term, adjs in top_adjs_per_term.items()
            for adj in adjs
        ]
    )
    logodds_selected = logodds_df.merge(selected_pairs, on=["term", "adjective"])
    gp_df = build_gp_df(logodds_selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        make_term_chart(logodds_df, gp_df, term, top_adjs_per_term[term])
        for term in top_adjs_per_term
    ]
    combined = alt.vconcat(*charts).resolve_scale(color="independent")
    out_path = OUTPUT_DIR / "gp_logodds.html"
    combined.save(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
