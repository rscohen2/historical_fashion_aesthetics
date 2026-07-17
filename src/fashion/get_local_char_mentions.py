"""
Generate noun mentions to use for adjective extraction, but use only the passage-local wearer model output.


The entities file should be a CSV file with the following columns:
    - mention_id (the character ID)
    - filename
    - sentence
    - term (the character)
    - start_idx
    - end_idx
    - sentence_start_idx
    - sentence_end_idx
    - gender

"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed


def gender(term):
    term = term.lower().strip()
    if term in set(["he", "him", "his"]):
        return "m"
    elif term in set(["she", "her", "hers"]):
        return "f"
    else:
        return "None"


def process_cooc_file(file, output_dir):
    results = []
    with open(file, "r") as f:
        for line in tqdm(f):
            obj = json.loads(line)

            characters = obj.get("characters", [])
            for character in characters:
                if character.get("wearing", False):
                    mention_id = character.get("coref")
                    filename = obj.get("book_id")
                    sentence = obj.get("excerpt_text")
                    term = character.get("text")
                    start_idx = character.get("character_start_idx")
                    end_idx = character.get("character_end_idx")
                    sentence_start_idx = obj.get("excerpt_start")
                    sentence_end_idx = obj.get("excerpt_end")
                    results.append(
                        {
                            "mention_id": mention_id,
                            "filename": filename,
                            "sentence": sentence,
                            "term": term,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "sentence_start_idx": sentence_start_idx,
                            "sentence_end_idx": sentence_end_idx,
                            "gender": term,
                        }
                    )

    output_file = output_dir / f"{file.stem}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")


def main(wearing_dir, output_dir, num_processes, concurrent_processes):
    def process(subset: list[Path]):
        for cooc_file in subset:
            process_cooc_file(cooc_file, output_dir)

    run_distributed(
        process,
        sorted(list(wearing_dir.glob("*.ndjson"))),
        script_path=__file__,
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=[
            "--wearing_dir",
            str(wearing_dir),
            "--output_dir",
            str(output_dir),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_distributed_args(parser)
    parser.add_argument("--wearing_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    main(
        args.wearing_dir,
        args.output_dir,
        args.num_processes,
        args.concurrent_processes,
    )
