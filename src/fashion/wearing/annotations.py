"""
Calculate agreement for annotation data.

Also used to contain some code for loading annotations.
"""

import json
from re import I
import numpy as np
import krippendorff
from collections import defaultdict
from typing import List, Dict, Tuple, Set


def load_and_filter_data(filepath: str) -> Dict[str, List[Dict]]:
    """
    Load NDJSON data and organize by datum_id.
    Returns a dictionary mapping datum_id to list of annotations.
    """
    data_by_datum = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("is_complete", False):
                datum_id = str(entry["datum_id"])
                data_by_datum[datum_id].append(entry)
    return data_by_datum


def explode_annotation(annotation: dict, datum: dict) -> list[bool]:
    """
    Convert the annotation to a list of booleans, where each boolean represents
    if an entity span is wearing the fashion item.
    """
    # TODO: handle negation
    annotation_corefs = set([char[0] for char in annotation["characters"]])
    return [char["coref"] in annotation_corefs for char in datum["characters"]]


def get_annotation_data(filepath: str) -> List[Dict]:
    """
    Load NDJSON data and return a list of annotations, where each datum is
    represented once and has a completed annotation. If there are multiple,
    discard any where there is a disagreement.
    """
    data_by_datum = load_and_filter_data(filepath)
    data = []
    for datum_id, annotations in data_by_datum.items():
        if len(annotations) > 1 and not all(
            annotation["annotation"]["characters"]
            == annotations[0]["annotation"]["characters"]
            for annotation in annotations
        ):
            continue
        if annotations[0]["annotation"]["characters"] is None:
            annotations[0]["annotation"]["characters"] = []

        # Handle some malformed data by dropping any entity mentions that are out of bounds
        # TODO: fix this
        annotations[0]["datum"]["characters"] = [
            char
            for char in annotations[0]["datum"]["characters"]
            if char["character_start_idx"] - annotations[0]["datum"]["excerpt_start"]
            >= 0
            and char["character_end_idx"] - annotations[0]["datum"]["excerpt_start"]
            <= len(annotations[0]["datum"]["excerpt_text"])
        ]
        if len(annotations[0]["datum"]["characters"]) == 0:
            continue

        data.append(annotations[0])
    print(f"Loaded {len(data)} data points")
    return data


def get_annotation_splits(data: List[Dict]) -> List[List[Dict]]:
    """
    Split the data into train/dev/test in a 80/10/10 split.
    Should stratify by book_id.
    """
    # get all books
    books = sorted(list(set([entry["datum"]["book_id"] for entry in data])))
    rng = np.random.default_rng(42)

    # get 80% of books for train
    train_books = set(rng.choice(books, size=int(len(books) * 0.8), replace=False))

    # get 10% of books for dev
    dev_books = set(
        rng.choice(
            sorted(list(set(books) - train_books)),
            size=int(len(books) * 0.1),
            replace=False,
        )
    )

    # get 10% of books for test
    test_books = set(books) - train_books - dev_books

    # split data into splits
    train_data = sorted(
        [entry for entry in data if entry["datum"]["book_id"] in train_books],
        key=lambda x: x["datum_id"],
    )
    dev_data = sorted(
        [entry for entry in data if entry["datum"]["book_id"] in dev_books],
        key=lambda x: x["datum_id"],
    )
    test_data = sorted(
        [entry for entry in data if entry["datum"]["book_id"] in test_books],
        key=lambda x: x["datum_id"],
    )

    return [train_data, dev_data, test_data]


def prepare_agreement_data(data: dict[str, list[dict]]) -> dict[str, list[bool]]:
    data_by_annotator: dict[str, list[bool]] = defaultdict(list)
    for _, entries in data.items():
        if len(entries) != 2:
            continue
        for entry in entries:
            data_by_annotator[str(entry["user_id"])].extend(
                explode_annotation(entry["annotation"], entry["datum"])
            )
    return data_by_annotator


def calculate_krippendorff_alpha(
    datum_user_annotations: dict[str, list[bool]],
) -> float:
    """
    Calculate Krippendorff's alpha for the multilabel annotations.
    """
    # Create reliability data matrix
    reliability_data = [annotations for annotations in datum_user_annotations.values()]

    # Convert to numpy array for krippendorff
    reliability_data = np.array(reliability_data).astype(int)

    # Calculate alpha
    alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
    return alpha


def main():
    # Load and process data
    data_by_datum = load_and_filter_data("data/wearing_data.ndjson")
    datum_user_annotations = prepare_agreement_data(data_by_datum)

    # Calculate agreement
    alpha = calculate_krippendorff_alpha(datum_user_annotations)

    print(f"Krippendorff's alpha: {alpha:.3f}")
    print(f"Number of users: {len(datum_user_annotations)}")
    print(f"Number of complete data points: {len(data_by_datum)}")


if __name__ == "__main__":
    main()
