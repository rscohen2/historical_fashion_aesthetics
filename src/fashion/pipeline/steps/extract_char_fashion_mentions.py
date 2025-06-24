"""Information about co-occurrence of fashion items with character entity mentions."""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed
from fashion.paths import DATA_DIR
from fashion.sources import Source, add_source_argument
from fashion.span_utils import get_k_closest_spans


def process_books(
    book_ids, fashion_mention_file, booknlp_dir, data_source, output_dir, rank=0
):
    fashion_mentions = pd.read_csv(fashion_mention_file)
    fashion_mentions = fashion_mentions.set_index("filename")

    def fashion_character_cooc(book_id):
        # get fashion terms
        text_filename = f"{book_id}.txt"
        fashion_mentions_book = fashion_mentions.loc[text_filename]
        # load book tokens
        tokens_book = pd.read_csv(
            booknlp_dir / f"{book_id}/{book_id}.tokens",
            sep="\t",
            quoting=3,  # 3 = csv.QUOTE_NONE
        ).set_index("token_ID_within_document")
        # get character mentions
        character_mentions_book = pd.read_csv(
            booknlp_dir / f"{book_id}/{book_id}.entities", sep="\t", quoting=3
        )
        # merge character mentions with token offsets
        character_mentions_book = character_mentions_book.merge(
            tokens_book[["byte_onset"]],
            left_on="start_token",
            right_index=True,
        )
        character_mentions_book = character_mentions_book.merge(
            tokens_book[["byte_offset"]],
            left_on="end_token",
            right_index=True,
        )
        character_mentions_book = character_mentions_book.rename(
            columns={
                "byte_onset": "character_start_idx",
                "byte_offset": "character_end_idx",
            }
        )

        book_text = data_source.load_text(book_id).text

        results = []

        for start_idx, end_idx, term, mention_id in zip(
            fashion_mentions_book.start_idx + fashion_mentions_book.sentence_start_idx,
            fashion_mentions_book.end_idx + fashion_mentions_book.sentence_start_idx,
            fashion_mentions_book.term,
            fashion_mentions_book.mention_id,
        ):
            # get the fashion term span
            fashion_span = (start_idx, end_idx)

            closest_character_inds, closest_character_dists = get_k_closest_spans(
                fashion_span,
                (
                    character_mentions_book.character_start_idx.tolist(),
                    character_mentions_book.character_end_idx.tolist(),
                ),
                k=10,
            )

            # find sentence idx bounds that a token is in
            def get_sentence_bounds(token_id):
                sentence_id = tokens_book.loc[token_id].sentence_ID
                start_idx = tokens_book[
                    tokens_book.sentence_ID == sentence_id
                ].byte_onset.min()
                end_idx = tokens_book[
                    tokens_book.sentence_ID == sentence_id
                ].byte_offset.max()

                return start_idx, end_idx

            # # fmt: off
            # import ipdb; ipdb.set_trace()  # noqa: E702
            # # fmt: on

            excerpt_start = max(
                start_idx - 500,
                get_sentence_bounds(
                    character_mentions_book.start_token.iloc[
                        closest_character_inds
                    ].min()
                )[0],
            )
            excerpt_end = min(
                end_idx + 500,
                get_sentence_bounds(
                    character_mentions_book.end_token.iloc[closest_character_inds].max()
                )[1],
            )
            if excerpt_end < start_idx or excerpt_start > end_idx:
                print(
                    f"Skipping {book_id} {term} ({start_idx}:{end_idx}) as the excerpt ({excerpt_start}:{excerpt_end}) does not overlap with the fashion term."
                )
                print(
                    f"Excerpt: {book_text[excerpt_start:excerpt_end]}, "
                    f"Fashion term: {book_text[start_idx:end_idx]}"
                )
                continue

            result = {
                "book_id": book_id,
                "mention_id": mention_id,
                "fashion_term": term,
                "fashion_start_idx": start_idx,
                "fashion_end_idx": end_idx,
                "excerpt_start": int(excerpt_start),
                "excerpt_end": int(excerpt_end),
                "excerpt_text": book_text[excerpt_start:excerpt_end],
                "characters": [
                    {
                        "character_start_idx": int(
                            character_mentions_book.character_start_idx.iloc[i]
                        ),
                        "character_end_idx": int(
                            character_mentions_book.character_end_idx.iloc[i]
                        ),
                        "coref": str(character_mentions_book["COREF"].iloc[i]),
                        "text": str(character_mentions_book.text.iloc[i]),
                        "distance": int(dist),
                    }
                    for i, dist in zip(closest_character_inds, closest_character_dists)
                ],
            }
            results.append(result)

        return results

    results = []
    for bookid in tqdm(book_ids, desc="Processing books"):
        if not (booknlp_dir / f"{bookid}/{bookid}.entities").exists():
            print(f"Skipping {bookid} as it has no entities file.")
            continue
        try:
            results.extend(fashion_character_cooc(bookid))
        except Exception as e:
            print(f"Error processing {bookid}: {e}")
            continue

    # write as ndjson
    output_file = output_dir / f"cooc.{rank}.ndjson"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"{json.dumps(result)}\n")


def main(
    source: Source,
    fashion_mention_file: Path,
    booknlp_dir: Path,
    output_dir: Path,
    num_processes: int,
    concurrent_processes: int,
):
    def process(subset: list[str]):
        process_books(subset, fashion_mention_file, booknlp_dir, source, output_dir)

    run_distributed(
        process,
        sorted(list(source.iter_book_ids())),
        script_path=__file__,
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=[
            "--source",
            str(source.name.lower()),
            "--fashion_mention_file",
            str(fashion_mention_file),
            "--booknlp_dir",
            str(booknlp_dir),
            "--output_dir",
            str(output_dir),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract co-occurrence of fashion items with character mentions in books."
    )
    add_source_argument(parser)
    add_distributed_args(parser)
    parser.add_argument(
        "--fashion_mention_file",
        type=Path,
        default=DATA_DIR / "filtered_fashion_texts.csv",
        help="CSV file containing fashion mentions.",
    )
    parser.add_argument(
        "--booknlp_dir",
        type=Path,
        default=DATA_DIR / "booknlp",
        help="Directory containing the booknlp processed data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "fashion_character_cooc",
        help="Directory to save the output co-occurrence data.",
    )
    args = parser.parse_args()

    main(
        source=args.source,
        fashion_mention_file=args.fashion_mention_file,
        booknlp_dir=args.booknlp_dir,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        concurrent_processes=args.concurrent_processes,
    )
