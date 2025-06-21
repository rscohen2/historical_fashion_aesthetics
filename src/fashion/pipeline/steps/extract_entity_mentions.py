"""
Given an annotated cooc file and the processed booknlp files, extract
the sentence which contains the entity mention, based on coref.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from fashion.distributed import add_distributed_args, run_distributed
from fashion.sources import Source, add_source_argument


def load_cooc_data(cooc_file: Path) -> list[dict]:
    with open(cooc_file, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def process_cooc_file(
    cooc_file: Path, booknlp_dir: Path, output_dir: Path, source: Source
):
    # load fashion coocs
    fashion_coocs = load_cooc_data(cooc_file)
    output_file = output_dir / f"{cooc_file.stem}.csv"

    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping.")
        return

    # get a list of entities that are associated with fashion for each book
    fashion_entities = defaultdict(set)
    for cooc in fashion_coocs:
        book_id = cooc["book_id"]
        for character in cooc["characters"]:
            if character.get("wearing", False):
                fashion_entities[book_id].add(character["coref"])

    # iterate through the books
    for book_id, entity_corefs in fashion_entities.items():
        tokens_book = pd.read_csv(
            booknlp_dir / f"{book_id}/{book_id}.tokens",
            sep="\t",
            quoting=3,  # 3 = csv.QUOTE_NONE
        ).set_index("token_ID_within_document")

        # calculate sentence start and end indices
        tokens_book["sentence_start"] = tokens_book.groupby(
            "sentence_ID"
        ).byte_onset.transform("min")
        tokens_book["sentence_end"] = tokens_book.groupby(
            "sentence_ID"
        ).byte_offset.transform("max")

        tokens_book["sentence_has_adjectives"] = tokens_book.groupby(
            "sentence_ID"
        ).POS_tag.transform(lambda x: x.eq("ADJ").any())

        # get character mentions
        entities_book = pd.read_csv(
            booknlp_dir / f"{book_id}/{book_id}.entities", sep="\t", quoting=3
        )
        # merge character mentions with token offsets
        entities_book = entities_book.merge(
            tokens_book[["byte_onset", "sentence_start", "sentence_has_adjectives"]],
            left_on="start_token",
            right_index=True,
        )
        entities_book = entities_book.merge(
            tokens_book[["byte_offset", "sentence_end"]],
            left_on="end_token",
            right_index=True,
        )
        entities_book = entities_book.rename(
            columns={
                "byte_onset": "entity_start_idx",
                "byte_offset": "entity_end_idx",
            }
        )

        entities_book = entities_book[
            entities_book.COREF.astype(str).isin(entity_corefs)
            & entities_book.sentence_has_adjectives
            & entities_book.cat.eq("PER")
        ]
        # filename,sentence,term,start_idx,end_idx,sentence_start_idx,sentence_end_idx

        book = source.load_text(book_id)

        entities_book["sentence"] = entities_book[
            ["sentence_start", "sentence_end"]
        ].apply(lambda x: book.text[x.sentence_start : x.sentence_end], axis=1)

        entities_book["term"] = entities_book[
            ["entity_start_idx", "entity_end_idx"]
        ].apply(lambda x: book.text[x.entity_start_idx : x.entity_end_idx], axis=1)

        entities_book = entities_book.drop_duplicates(subset=["COREF", "sentence"])

        output_df = pd.DataFrame(
            {
                "mention_id": entities_book.COREF,
                "filename": book_id,
                "sentence": entities_book.sentence,
                "term": entities_book.term,
                "start_idx": entities_book.entity_start_idx
                - entities_book.sentence_start,
                "end_idx": entities_book.entity_end_idx - entities_book.sentence_start,
                "sentence_start_idx": entities_book.sentence_start,
                "sentence_end_idx": entities_book.sentence_end,
            }
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_df.to_csv(
                output_file,
                index=False,
                mode="a",
                header=False,
                encoding="utf-8",
            )
        else:
            output_df.to_csv(
                output_file,
                index=False,
                header=True,
                mode="w",
                encoding="utf-8",
            )


def main(
    source, wearing_dir, booknlp_dir, output_dir, num_processes, concurrent_processes
):
    def process(subset: list[Path]):
        for cooc_file in subset:
            process_cooc_file(cooc_file, booknlp_dir, output_dir, source)

    run_distributed(
        process,
        sorted(list(wearing_dir.glob("*.ndjson"))),
        script_path=__file__,
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=[
            "--source",
            str(source.name.lower()),
            "--wearing_dir",
            str(wearing_dir),
            "--booknlp_dir",
            str(booknlp_dir),
            "--output_dir",
            str(output_dir),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_argument(parser)
    add_distributed_args(parser)
    parser.add_argument("--wearing_dir", type=Path)
    parser.add_argument("--booknlp_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    main(
        args.source,
        args.wearing_dir,
        args.booknlp_dir,
        args.output_dir,
        args.num_processes,
        args.concurrent_processes,
    )
