"""
Use the following files as input:
- `adjectives/fashion/*.csv`
- `adjectives/entities/*.csv`
- `wearing/*.ndjson`

Generate two files:

- `final/fashion_mentions.parquet`
- `final/characters.parquet`

Each row in fashion_mentions.parquet should contain:
- book_id
- character_id
- mention_id
- term
- start_idx  (within excerpt)
- end_idx  (within excerpt)
- sentence
- sentence_start_idx  (within book)
- sentence_end_idx  (within book)
- adjectives  (list of adjectives for the fashion item)

Each row in characters.parquet should contain:
- book_id
- character_id
- character_adjectives  (list of adjectives for the character)
- character_start_idx  (list, within book)
- character_end_idx  (list, within book)
"""

import argparse
from pathlib import Path

import pandas as pd


def read_adjective_files(adjective_dir: Path) -> pd.DataFrame:
    adjective_files = adjective_dir.glob("*.csv")
    dfs = []
    for adjective_file in adjective_files:
        adjective_df = pd.read_csv(adjective_file)
        dfs.append(adjective_df)
    return pd.concat(dfs)


def read_wearing_files(wearing_dir: Path) -> pd.DataFrame:
    wearing_files = wearing_dir.glob("*.ndjson")
    dfs = []
    for wearing_file in wearing_files:
        wearing_df = pd.read_json(wearing_file, lines=True)
        dfs.append(wearing_df)
    return pd.concat(dfs)


def create_character_df(entities_adjectives: pd.DataFrame) -> pd.DataFrame:
    entities_adjectives["character_start_idx"] = (
        entities_adjectives["mention_start_idx"]
        + entities_adjectives["sentence_start_idx"]
    )
    entities_adjectives["character_end_idx"] = (
        entities_adjectives["mention_end_idx"]
        + entities_adjectives["sentence_start_idx"]
    )

    grouped = (
        entities_adjectives.groupby(["filename", "mention_id"])
        .agg(
            {
                "adjective": list,
                "character_start_idx": list,
                "character_end_idx": list,
            }
        )
        .reset_index()
    )

    grouped.rename(
        columns={
            "filename": "book_id",
            "mention_id": "character_id",
            "adjective": "adjectives",
        },
        inplace=True,
    )

    return grouped


def create_fashion_df(
    wearing_df: pd.DataFrame,
    fashion_adjectives: pd.DataFrame,
    fashion_mentions: pd.DataFrame,
) -> pd.DataFrame:
    fashion_mentions["book_id"] = fashion_mentions["filename"].str.split(".").str[0]
    fashion_mentions.set_index(["book_id", "mention_id"], inplace=True)
    fashion_mentions = fashion_mentions[
        [
            "term",
            "sentence",
            "start_idx",
            "end_idx",
            "sentence_start_idx",
            "sentence_end_idx",
        ]
    ]

    fashion_adjectives["book_id"] = fashion_adjectives["filename"].str.split(".").str[0]
    fashion_adjectives.set_index(["book_id", "mention_id"], inplace=True)
    fashion_adjectives = (
        fashion_adjectives.groupby(["book_id", "mention_id"])
        .agg({"adjective": list})
        .rename(columns={"adjective": "adjectives"})
    )

    wearing_df.set_index(["book_id", "mention_id"], inplace=True)
    wearing_df = (
        wearing_df.characters.apply(
            lambda x: set([c["coref"] for c in x if "wearing" in c and c["wearing"]])
        )
        .explode()
        .to_frame("character_id")
    )

    merged = fashion_mentions.merge(
        fashion_adjectives, left_index=True, right_index=True, how="outer"
    ).merge(wearing_df, left_index=True, right_index=True, how="outer")

    # Hack: https://stackoverflow.com/a/64207857
    merged["adjectives"] = merged["adjectives"].fillna("").apply(list)
    merged = merged.reset_index()

    return merged


def finalize_outputs(
    fashion_mention_file: Path,
    adjective_dir: Path,
    wearing_dir: Path,
    output_dir: Path,
):
    fashion_mentions = pd.read_csv(fashion_mention_file)
    fashion_adjectives = read_adjective_files(adjective_dir / "fashion")
    entities_adjectives = read_adjective_files(adjective_dir / "entities")
    wearing_df = read_wearing_files(wearing_dir)

    character_df = create_character_df(entities_adjectives)
    fashion_df = create_fashion_df(wearing_df, fashion_adjectives, fashion_mentions)

    output_dir.mkdir(parents=True, exist_ok=True)
    character_df.to_parquet(output_dir / "final" / "characters.parquet", index=False)
    fashion_df.to_parquet(
        output_dir / "final" / "fashion_mentions.parquet", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashion_mention_file", type=Path, required=True)
    parser.add_argument("--adjective_dir", type=Path, required=True)
    parser.add_argument("--wearing_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    finalize_outputs(
        fashion_mention_file=args.fashion_mention_file,
        adjective_dir=args.adjective_dir,
        wearing_dir=args.wearing_dir,
        output_dir=args.output_dir,
    )
