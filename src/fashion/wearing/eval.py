import itertools
import argparse
import json

from tqdm import tqdm
from sklearn.metrics import classification_report

from fashion.wearing import Wearing, WearingLLM
from fashion.wearing.annotations import (
    explode_annotation,
    get_annotation_data,
    get_annotation_splits,
)


def prepare_batch(
    batch_data: list[dict],
) -> tuple[
    list[str], list[tuple[int, int]], list[list[tuple[int, int]]], list[list[str]]
]:
    batch_texts = [datum["excerpt_text"] for datum in batch_data]
    batch_fashion_spans = [
        (
            datum["fashion_start_idx"] - datum["excerpt_start"],
            datum["fashion_end_idx"] - datum["excerpt_start"],
        )
        for datum in batch_data
    ]
    batch_entity_spans = [
        [
            (
                char["character_start_idx"] - datum["excerpt_start"],
                char["character_end_idx"] - datum["excerpt_start"],
            )
            for char in datum["characters"]
        ]
        for datum in batch_data
    ]
    batch_coref_labels = get_batch_coref_labels(batch_data)
    return batch_texts, batch_fashion_spans, batch_entity_spans, batch_coref_labels


def get_batch_coref_labels(batch_data: list[dict]) -> list[list[str]]:
    coref_labels = []
    for datum in batch_data:
        coref_set = set([char["coref"] for char in datum["characters"]])
        coref_idx = {coref: str(i) for i, coref in enumerate(coref_set)}
        coref_labels.append([coref_idx[char["coref"]] for char in datum["characters"]])
    return coref_labels


def label_batch(
    batch_data: list[dict], wearing_results: list[list[bool]]
) -> list[dict]:
    labeled_batch = []
    for datum, wearing_result in zip(batch_data, wearing_results):
        for char, wearing_label in zip(datum["characters"], wearing_result):
            char["wearing"] = wearing_label
        labeled_batch.append(datum)

    return labeled_batch


def evaluate(data: list[dict], wearing_model: Wearing, batch_size: int = 10):
    output_data = []
    y_true = []
    y_pred = []
    for batch in tqdm(
        itertools.batched(data, batch_size), total=len(data) // batch_size
    ):
        batch_data = [entry["datum"] for entry in batch]
        batch_annotations = [
            explode_annotation(entry["annotation"], entry["datum"]) for entry in batch
        ]

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

        y_pred.extend([wearing for result in wearing_results for wearing in result])
        y_true.extend([wearing for result in batch_annotations for wearing in result])

    print(classification_report(y_true, y_pred))
    return output_data


def choose_model(model_name: str) -> Wearing:
    if model_name == "WearingLLM":
        return WearingLLM()
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wearing-model", type=str, default="WearingLLM", choices=["WearingLLM"]
    )
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    train, dev, test = get_annotation_splits(
        get_annotation_data("data/wearing_data.ndjson")
    )

    wearing = choose_model(args.wearing_model)
    output_data = evaluate(test, wearing, args.batch_size)
    with open(
        f"data/evaluation/inference_{args.wearing_model.lower()}.ndjson", "w"
    ) as f:
        for datum in output_data:
            f.write(json.dumps(datum) + "\n")
