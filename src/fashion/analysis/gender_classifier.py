"""Train per-decade gender classifier and save fashion term coefficients.

Trains a logistic regression to predict character gender (male/female) from
fashion terms, fitting one model per decade. The coefficient for each term
reflects how strongly it indexes male (positive) or female (negative) gender.

Output: data/analysis/gender_classifier/coefficients.parquet
        data/analysis/gender_classifier/coefficients.tsv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from fashion.paths import DATA_DIR

EXPERIMENT = "gender_classifier"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
DECADES = list(range(180, 192))
BINARY_GENDERS = {"male", "female"}


def build_training_data(
    df: pd.DataFrame,
) -> tuple[list[str], list[list[str]], np.ndarray]:
    """Return (genders, features, decades) for mentions with a binary gender and ≥1 fashion adjective."""
    genders, features, decades = [], [], []
    subset = df.dropna(subset=["gender_dep_pron"])
    for row in tqdm(subset.itertuples(), total=len(subset)):
        if row.gender_dep_pron not in BINARY_GENDERS:
            continue
        adjs = [adj for adj in row.adjectives_fashion if adj is not None]
        if not adjs:
            continue
        genders.append(row.gender_dep_pron)
        features.append([f"term_{row.term.lower()}"])
        decades.append(row.decade)
    return genders, features, np.array(decades)


def train_models(
    genders: list[str],
    features: list[list[str]],
    decades: np.ndarray,
    min_df: int = 100,
) -> tuple[CountVectorizer, LabelEncoder, dict[int, LogisticRegression]]:
    encoder = CountVectorizer(
        input="content",
        lowercase=False,
        preprocessor=None,
        tokenizer=lambda x: x,
        min_df=min_df,
    )
    encoder.fit(features)

    le = LabelEncoder()
    le.fit(genders)

    models: dict[int, LogisticRegression] = {}
    for decade in DECADES:
        idx = np.where(decades == decade)[0]
        if len(idx) == 0:
            continue
        X = encoder.transform([features[i] for i in idx])
        y = le.transform([genders[i] for i in idx])
        model = LogisticRegression(penalty="l1", solver="liblinear")
        model.fit(X, y)
        models[decade] = model
        print(f"Decade {int(decade * 10)}s: trained on {len(idx):,} examples")

    return encoder, le, models


def build_coefficients_df(
    encoder: CountVectorizer,
    le: LabelEncoder,
    models: dict[int, LogisticRegression],
) -> pd.DataFrame:
    """Long-form DataFrame of (decade, term, coefficient).

    Coefficient is oriented so positive = male-indexing, negative = female-indexing.
    """
    feature_names = encoder.get_feature_names_out()
    # LabelEncoder sorts classes alphabetically: classes_[0]="female", classes_[1]="male"
    # coef_[0] > 0 predicts classes_[1] = "male", so sign is already correct.
    sign = 1 if list(le.classes_).index("male") == 1 else -1

    rows = []
    for decade, model in models.items():
        for term, coef in zip(feature_names, model.coef_[0]):
            rows.append(
                {
                    "decade": decade,
                    "term": term.removeprefix("term_"),
                    "coefficient": float(sign * coef),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=100,
        help="Minimum document frequency for the vocabulary (default: 100).",
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
    genders, features, decades = build_training_data(df)
    encoder, le, models = train_models(genders, features, decades, args.min_df)

    coef_df = build_coefficients_df(encoder, le, models)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_df.to_parquet(OUTPUT_DIR / "coefficients.parquet", index=False)
    coef_df.to_csv(OUTPUT_DIR / "coefficients.tsv", sep="\t", index=False)

    n_terms = coef_df["term"].nunique()
    n_decades = coef_df["decade"].nunique()
    print(f"Saved coefficients for {n_terms} terms across {n_decades} decades")
    print(f"  → {OUTPUT_DIR / 'coefficients.parquet'}")
    print(f"  → {OUTPUT_DIR / 'coefficients.tsv'}")


if __name__ == "__main__":
    main()
