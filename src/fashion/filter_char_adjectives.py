"""
Stanza coref was too lenient, so we need to filter out the adjectives that
aren't actually related to the character by using the character coref output
from booknlp.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed
from fashion.span_utils import (
    contained_by,
    contains,
    get_fn_rows,
    get_overlap_rows,
    max_padding,
    padding,
)


def process_book(
    book_id: str, booknlp_dir: Path, adjectives: pd.DataFrame, invert: bool
) -> pd.DataFrame:
    booknlp_dir = booknlp_dir / book_id
    booknlp_entities_file = booknlp_dir / f"{book_id}.entities"
    booknlp_tokens_file = booknlp_dir / f"{book_id}.tokens"

    try:
        booknlp_entities = pd.read_csv(booknlp_entities_file, sep="\t", quoting=3)
        booknlp_tokens = pd.read_csv(
            booknlp_tokens_file, sep="\t", quoting=3
        ).set_index("token_ID_within_document")
    except FileNotFoundError:
        return pd.DataFrame()

    booknlp_tokens["sentence_start"] = booknlp_tokens.groupby(
        "sentence_ID"
    ).byte_onset.transform("min")
    booknlp_tokens["sentence_end"] = booknlp_tokens.groupby(
        "sentence_ID"
    ).byte_offset.transform("max")

    # merge character mentions with token offsets
    booknlp_entities = booknlp_entities.merge(
        booknlp_tokens[["byte_onset", "sentence_start"]],
        left_on="start_token",
        right_index=True,
    )
    booknlp_entities = booknlp_entities.merge(
        booknlp_tokens[["byte_offset", "sentence_end"]],
        left_on="end_token",
        right_index=True,
    )
    booknlp_entities = booknlp_entities.rename(
        columns={
            "byte_onset": "entity_start_idx",
            "byte_offset": "entity_end_idx",
        }
    )
    booknlp_entities = booknlp_entities[booknlp_entities.cat.eq("PER")]

    entity_span_starts = booknlp_entities["entity_start_idx"]
    entity_span_ends = booknlp_entities["entity_end_idx"]

    adjective_noun_starts = (
        adjectives["sentence_start_idx"] + adjectives["noun_start_idx"]
    )
    adjective_noun_ends = adjectives["sentence_start_idx"] + adjectives["noun_end_idx"]

    overlaps = get_fn_rows(
        (adjective_noun_starts, adjective_noun_ends),
        (entity_span_starts, entity_span_ends),
        contained_by,
    )

    has_overlaps = [bool(overlap) ^ invert for overlap in overlaps]

    return pd.DataFrame(adjectives[has_overlaps])


def main(
    adjectives_dir: Path,
    booknlp_dir: Path,
    output_dir: Path,
    invert: bool,
    num_processes,
    concurrent_processes,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(adj_files: list[Path]):
        rank = int(os.environ.get("DISTRIBUTED_RANK", 0))
        all_dfs = []
        for adj_file in tqdm(adj_files):
            adjectives = pd.read_csv(adj_file)
            books = adjectives.filename.unique()
            for book_id in tqdm(books):
                adj_subset = pd.DataFrame(adjectives[adjectives.filename == book_id])
                if book_id[-4:] == ".txt":
                    book_id = book_id[:-4]
                adjectives_filtered = process_book(
                    book_id, booknlp_dir, adj_subset, invert
                )

                all_dfs.append(adjectives_filtered)

        all_dfs = pd.concat(all_dfs)
        all_dfs.to_csv(output_dir / f"filtered.{rank}.csv", index=False)

    all_adj_files = sorted(adjectives_dir.glob("**/*.csv"))
    run_distributed(
        process,
        all_adj_files,
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=[
            "--adjectives_dir",
            str(adjectives_dir),
            "--booknlp_dir",
            str(booknlp_dir),
            "--output_dir",
            str(output_dir),
            "--invert" if invert else "",
        ],
        max_retries=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjectives_dir", type=Path, required=True)
    parser.add_argument("--booknlp_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--invert", action="store_true")
    add_distributed_args(parser)
    args = parser.parse_args()

    main(**vars(args))
