import argparse
import itertools
import json
from pathlib import Path

from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed, run_single
from fashion.wearing import Wearing, WearingBert
from fashion.wearing.eval import label_batch, prepare_batch


def load_cooc_data(fashion_cooc_file: Path) -> list[dict]:
    with open(fashion_cooc_file, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def inference(data: list[dict], wearing_model: Wearing, batch_size: int = -1):
    output_data = []
    y_pred = []
    if batch_size == -1:
        batch_size = len(data)
    for batch in tqdm(
        itertools.batched(data, batch_size), total=len(data) // batch_size
    ):
        if len(batch) == 0:
            continue
        batch_data = list(batch)

        batch_texts, batch_fashion_spans, batch_entity_spans, batch_coref_labels = (
            prepare_batch(batch_data)
        )

        wearing_results = wearing_model.is_wearing(
            batch_texts,
            batch_fashion_spans,
            batch_entity_spans,
            batch_coref_labels,
        )

        labeled_batch = label_batch(batch_data, wearing_results)

        output_data.extend(labeled_batch)

        y_pred.extend(
            [
                wearing
                for result in wearing_results
                for wearing in result
                if wearing is not None
            ]
        )

    return output_data


def main(
    fashion_cooc_dir: Path,
    output_dir: Path,
    num_processes: int,
    concurrent_processes: int,
):
    def process(subset: list[Path]):
        for fashion_cooc_file in subset:
            output_file = output_dir / f"{fashion_cooc_file.stem}.ndjson"
            if output_file.exists():
                continue
            data = load_cooc_data(fashion_cooc_file)
            wearing_model = WearingBert()
            output_data = inference(data, wearing_model)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                for entry in output_data:
                    f.write(json.dumps(entry) + "\n")

            # free up memory
            del wearing_model

    run_single(
        process,
        sorted(list(fashion_cooc_dir.glob("*.ndjson"))),
        script_path=__file__,
        # total_processes=num_processes,
        # concurrent_processes=concurrent_processes,
        extra_args=[
            "--fashion_cooc_dir",
            str(fashion_cooc_dir),
            "--output_dir",
            str(output_dir),
        ],
        # max_retries=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashion_cooc_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    add_distributed_args(parser)
    args = parser.parse_args()

    main(**vars(args))
