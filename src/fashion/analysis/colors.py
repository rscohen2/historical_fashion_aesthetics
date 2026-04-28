"""Plot color adjective prevalence for gendered fashion terms over time.

For each gender, uses classifier coefficients (from gender_classifier.py) to
identify the top fashion terms by gender-indexing strength, then plots the
relative prevalence of color adjectives used to describe each term across
decades, with binomial 95% confidence intervals. Each fashion item gets its
own facet panel.

Requires: data/analysis/gender_classifier/coefficients.parquet
Output:   data/analysis/colors/color_prevalence_{gender}.html
"""

import argparse
import itertools
import math
from pathlib import Path

import altair as alt
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.stats import norm

from fashion.paths import DATA_DIR

EXPERIMENT = "colors"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
CLASSIFIER_OUTPUT = DATA_DIR / "analysis" / "gender_classifier" / "coefficients.parquet"
DECADES = list(range(180, 192))


def build_color_set() -> set[str]:
    chromatic = wn.synset("chromatic.a.03")
    achromatic = wn.synset("achromatic.a.01")
    value = wn.synset("value.n.04")
    assert chromatic is not None and achromatic is not None and value is not None
    words = chromatic.similar_tos() + achromatic.similar_tos() + value.attributes()
    return set(
        itertools.chain.from_iterable(
            [lemma.name() for lemma in word.lemmas()] for word in words
        )
    )


def binom_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    p = k / n if n > 0 else 0.0
    z = float(norm.ppf(1 - (1 - conf) / 2))
    stderr = math.sqrt(max(0.0, p * (1 - p) / n)) if n > 0 else 0.0
    return max(0.0, p - z * stderr), min(1.0, p + z * stderr)


def get_top_terms(coef_df: pd.DataFrame, gender: str, top_n: int) -> list[str]:
    """Return terms in the top N for the given gender in at least one decade.

    Positive coefficient = male-indexing; negative = female-indexing.
    """
    ascending = gender == "female"
    top_per_decade = (
        coef_df.sort_values("coefficient", ascending=ascending)
        .groupby("decade")
        .head(top_n)
    )
    return top_per_decade["term"].unique().tolist()


def compute_color_prevalence(
    df: pd.DataFrame, colors: set[str], top_n_colors: int = 10
) -> pd.DataFrame:
    """Return a long-form DataFrame of color proportions per (decade, term).

    Proportions are relative to total term mentions so that the "(none)" row
    captures how often the term appears without any color description.
    """
    # One sample per (book, term) to avoid within-book repetition
    sampled = df.groupby(["book_id", "term"]).first().reset_index()
    sampled = sampled[sampled.decade.isin(DECADES)].copy()
    sampled["term"] = sampled["term"].str.lower()

    # Denominator: total mentions per (decade, term)
    totals = sampled.groupby(["decade", "term"]).size().reset_index(name="total")

    # Explode adjectives and keep only color words; deduplicate so each
    # (book, term, color) pair is counted at most once per mention.
    adj_df = sampled.explode("adjectives_fashion").copy()
    adj_df = adj_df[adj_df["adjectives_fashion"].str.lower().isin(colors)]
    adj_df["adjectives_fashion"] = adj_df["adjectives_fashion"].str.lower()
    adj_df = adj_df.drop_duplicates(subset=["book_id", "term", "adjectives_fashion"])

    counts = (
        adj_df.groupby(["decade", "term", "adjectives_fashion"])
        .size()
        .reset_index(name="count")
    )
    counts = counts.merge(totals, on=["decade", "term"])
    counts["proportion"] = counts["count"] / counts["total"]

    ci = counts.apply(lambda row: binom_ci(row["count"], row["total"]), axis=1)
    counts["ci_low"] = [v[0] for v in ci]
    counts["ci_high"] = [v[1] for v in ci]

    # Restrict to top N colors per term, then always append "(none)"
    top_colors = (
        counts.groupby(["term", "adjectives_fashion"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .groupby("term")
        .head(top_n_colors)[["term", "adjectives_fashion"]]
    )
    counts = counts.merge(top_colors, on=["term", "adjectives_fashion"])

    # "(none)": mentions where no color adjective was used
    colored = (
        adj_df.drop_duplicates(subset=["book_id", "term"])
        .groupby(["decade", "term"])
        .size()
        .reset_index(name="colored_count")
    )
    none_rows = totals.merge(colored, on=["decade", "term"], how="left")
    none_rows["colored_count"] = none_rows["colored_count"].fillna(0)
    none_rows["count"] = (none_rows["total"] - none_rows["colored_count"]).astype(int)
    none_rows["adjectives_fashion"] = "(none)"
    none_rows["proportion"] = none_rows["count"] / none_rows["total"]
    ci = none_rows.apply(lambda row: binom_ci(row["count"], row["total"]), axis=1)
    none_rows["ci_low"] = [v[0] for v in ci]
    none_rows["ci_high"] = [v[1] for v in ci]
    none_rows = none_rows[counts.columns]

    return pd.concat([counts, none_rows], ignore_index=True)


def make_chart(
    color_counts: pd.DataFrame, gender: str, terms: list[str]
) -> alt.FacetChart:
    """Faceted chart with one panel per fashion item, color encoding per color adjective."""
    data = color_counts[color_counts.term.isin(terms)]

    selection = alt.selection_point(
        fields=["adjectives_fashion"], bind="legend", on="click"
    )
    y_zoom = alt.selection_interval(encodings=["y"], resolve="global")

    line = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("decade:O", title="Decade", sort=DECADES),
            y=alt.Y("proportion:Q", title="Proportion of mentions"),
            color=alt.Color("adjectives_fashion:N", title="Color"),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
            tooltip=[
                "decade:O",
                "term:N",
                "adjectives_fashion:N",
                "proportion:Q",
                "ci_low:Q",
                "ci_high:Q",
            ],
        )
        .add_params(selection, y_zoom)
        .transform_filter(y_zoom)
    )
    band = (
        alt.Chart(data)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("decade:O", sort=DECADES),
            y=alt.Y("ci_low:Q"),
            y2=alt.Y2("ci_high:Q"),
            color=alt.Color("adjectives_fashion:N"),
            opacity=alt.condition(selection, alt.value(0.4), alt.value(0.05)),
        )
        .transform_filter(y_zoom)
    )

    return (
        (band + line)
        .facet(
            facet=alt.Facet("term:N", title="Fashion Item"),
            columns=3,
            title=f"Color Adjective Prevalence by Fashion Item ({gender.capitalize()}-gendered mentions)",
        )
        .resolve_scale(color="independent")
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
        "--gender",
        choices=["male", "female", "both"],
        default="both",
        help="Gender of mentions to include (default: both, one chart per gender).",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=5,
        help="Top N gender-indexing terms per decade from the classifier (default: 5).",
    )
    parser.add_argument(
        "--top-colors",
        type=int,
        default=10,
        help="Number of top color adjectives to show per term (default: 10).",
    )
    parser.add_argument(
        "--classifier-output",
        type=Path,
        default=CLASSIFIER_OUTPUT,
        help="Path to coefficients.parquet from gender_classifier.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.classifier_output.exists():
        raise FileNotFoundError(
            f"Classifier output not found: {args.classifier_output}\n"
            "Run `python -m fashion.analysis.gender_classifier` first."
        )

    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df = pd.read_parquet(input_path).reset_index()
    coef_df = pd.read_parquet(args.classifier_output)
    colors = build_color_set()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    genders = ["male", "female"] if args.gender == "both" else [args.gender]
    for gender in genders:
        terms = get_top_terms(coef_df, gender, args.top_terms)
        gender_df = df[df.gender_dep_pron.eq(gender)]
        color_counts = compute_color_prevalence(gender_df, colors, args.top_colors)

        term_totals = (
            color_counts[
                color_counts.term.isin(terms)
                & color_counts.adjectives_fashion.eq("(none)")
            ]
            .groupby("term")["total"]
            .sum()
            .sort_values(ascending=False)
        )
        print(f"\n{gender} terms (total mentions):")
        for term, n in term_totals.items():
            print(f"  {term}: {n:,}")

        chart = make_chart(color_counts, gender, terms)
        out_path = OUTPUT_DIR / f"color_prevalence_{gender}.html"
        chart.save(str(out_path))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
