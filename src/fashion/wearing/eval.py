"""
LLM gpt-4.1-nano
            precision    recall  f1-score   support

       False       0.84      0.96      0.90      1185
        True       0.95      0.81      0.87      1120

    accuracy                           0.89      2305
   macro avg       0.90      0.89      0.89      2305
weighted avg       0.90      0.89      0.89      2305

deberta-v3-base fine-tune
              precision    recall  f1-score   support

       False       0.93      0.89      0.91      1185
        True       0.89      0.93      0.91      1120

    accuracy                           0.91      2305
   macro avg       0.91      0.91      0.91      2305
weighted avg       0.91      0.91      0.91      2305

"""

import argparse
import itertools
import json

from sklearn.metrics import classification_report
from tqdm import tqdm

from fashion.wearing import Wearing, WearingBert, WearingLLM
from fashion.wearing.annotations import (
    explode_annotation,
    get_annotation_data,
    get_annotation_splits,
)


def in_bounds_characters(datum: dict) -> list[bool]:
    """Which of the datum's candidate characters lie inside its excerpt.

    The excerpt is clipped to 500 characters on either side of the fashion term, so
    some of the 10 nearest character mentions can fall outside it. Only in-bounds
    characters are shown to the wearing model.
    """
    return [
        char["character_start_idx"] - datum["excerpt_start"] >= 0
        and char["character_end_idx"] - datum["excerpt_start"]
        <= len(datum["excerpt_text"])
        for char in datum["characters"]
    ]


def prepare_batch(
    batch_data: list[dict],
) -> tuple[
    list[str], list[tuple[int, int]], list[list[tuple[int, int]]], list[list[str]]
]:
    """Inputs for Wearing.is_wearing. Entity spans and coref labels cover only the
    in-bounds characters (see in_bounds_characters), and label_batch maps the
    results back onto them."""
    batch_texts = [datum["excerpt_text"] for datum in batch_data]
    batch_fashion_spans = [
        (
            datum["fashion_start_idx"] - datum["excerpt_start"],
            datum["fashion_end_idx"] - datum["excerpt_start"],
        )
        for datum in batch_data
    ]
    batch_characters = [
        [
            char
            for char, in_bounds in zip(datum["characters"], in_bounds_characters(datum))
            if in_bounds
        ]
        for datum in batch_data
    ]
    batch_entity_spans = [
        [
            (
                char["character_start_idx"] - datum["excerpt_start"],
                char["character_end_idx"] - datum["excerpt_start"],
            )
            for char in characters
        ]
        for datum, characters in zip(batch_data, batch_characters)
    ]
    batch_coref_labels = get_batch_coref_labels(batch_characters)
    return batch_texts, batch_fashion_spans, batch_entity_spans, batch_coref_labels


def get_batch_coref_labels(batch_characters: list[list[dict]]) -> list[list[str]]:
    coref_labels = []
    for characters in batch_characters:
        coref_set = {char["coref"] for char in characters}
        coref_idx = {coref: str(i) for i, coref in enumerate(coref_set)}
        coref_labels.append([coref_idx[char["coref"]] for char in characters])
    return coref_labels


def label_batch(
    batch_data: list[dict], wearing_results: list[list[bool | None]]
) -> list[dict]:
    """Set every candidate character's "wearing" label. wearing_results has one
    label per in-bounds character, as prepare_batch passed them; out-of-bounds
    characters were never scored, so they are labeled False, as are skipped ones
    (None)."""
    labeled_batch = []
    for datum, wearing_result in zip(batch_data, wearing_results, strict=True):
        in_bounds = in_bounds_characters(datum)
        if len(wearing_result) != sum(in_bounds):
            raise ValueError(
                f"Got {len(wearing_result)} labels for {sum(in_bounds)} in-bounds "
                f"characters of mention {datum['mention_id']} in {datum['book_id']}"
            )
        wearing_labels = iter(wearing_result)
        for char, is_in_bounds in zip(datum["characters"], in_bounds):
            wearing_label = next(wearing_labels) if is_in_bounds else None
            char["wearing"] = wearing_label if wearing_label is not None else False
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
    elif model_name == "WearingBert":
        return WearingBert()
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wearing-model",
        type=str,
        default="WearingLLM",
        choices=["WearingLLM", "WearingBert"],
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
        f.writelines(json.dumps(datum) + "\n" for datum in output_data)
