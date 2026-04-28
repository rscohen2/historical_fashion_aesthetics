"""Analyze passages with fashion items that are described as black.

1. extract passages with fashion items that are described as black into
   `black_fashion_passages.tsv` along with their metadata (e.g., title, author,
   publication date, etc.)
2. extract a sample of 50 passages per decade into `black_fashion_passages_sample.tsv`
"""

import argparse

import pandas as pd

from fashion.paths import DATA_DIR

EXPERIMENT = "black_fashion"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract passages with fashion items described as black."
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        input_path = DATA_DIR / "sample_for_analysis.parquet"
    else:
        input_path = DATA_DIR / "final_for_analysis.parquet"

    df = pd.read_parquet(input_path)

    mask = df["adjectives_fashion"].apply(lambda x: "black" in x)
    black_df = df[mask].copy()
    black_df = black_df.reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "book_id",
        "character_id",
        "mention_id",
        "term",
        "adjectives_fashion",
        "sentence",
        "gender_dep_pron",
        "inferreddate",
        "decade_clean",
    ]
    black_df[output_cols].to_csv(
        OUTPUT_DIR / "black_fashion_passages.tsv", sep="\t", index=False
    )

    print(
        f"Extracted {len(black_df):,} passages to {OUTPUT_DIR / 'black_fashion_passages.tsv'}"
    )

    black_df.groupby("decade_clean").sample(50)[output_cols].to_csv(
        OUTPUT_DIR / "black_fashion_passages_sample.tsv", sep="\t", index=False
    )

    print(
        f"Extracted a sample of 50 passages per decade to {OUTPUT_DIR / 'black_fashion_passages_sample.tsv'}"
    )


if __name__ == "__main__":
    main()
