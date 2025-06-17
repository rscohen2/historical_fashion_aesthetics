"""
Calculate agreement for annotation data.

Also used to contain some code for loading annotations.
"""

import json
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


def prepare_agreement_data(
    data_by_datum: Dict[str, List[Dict]],
) -> Tuple[Dict[Tuple[str, str], List[Tuple[str, bool]]], Set[str]]:
    """
    Prepare data for agreement analysis.
    Returns a tuple of:
    1. Dictionary mapping (datum_id, user_id) to their character classifications
    2. Set of all unique tags
    Only includes data points where all users have provided annotations.
    """
    # First, find all unique users
    all_users = set()
    for annotations in data_by_datum.values():
        all_users.update(str(entry["user_id"]) for entry in annotations)

    # Filter for data points where all users have annotated
    complete_data = {
        datum_id: annotations
        for datum_id, annotations in data_by_datum.items()
        if len(annotations) == len(all_users)
    }

    # Organize annotations by datum and user
    datum_user_annotations = defaultdict(list)
    all_tags = set()

    for datum_id, annotations in complete_data.items():
        for entry in annotations:
            user_id = str(entry["user_id"])
            characters = entry["annotation"]["characters"]

            # Convert each character annotation to (tag, value) tuples
            for tag, value in characters:
                datum_user_annotations[(datum_id, user_id)].append((tag, value))
                all_tags.add(tag)

    return datum_user_annotations, all_tags


def calculate_krippendorff_alpha(
    datum_user_annotations: Dict[Tuple[str, str], List[Tuple[str, bool]]],
    all_tags: Set[str],
) -> float:
    """
    Calculate Krippendorff's alpha for the multilabel annotations.
    """
    # Create reliability data matrix
    reliability_data = []

    # Group annotations by datum_id
    datum_groups = defaultdict(list)
    for (datum_id, user_id), annotations in datum_user_annotations.items():
        datum_groups[datum_id].append((user_id, annotations))

    # For each datum, create a row in the reliability matrix
    for datum_id, user_annotations in datum_groups.items():
        # Sort by user_id to ensure consistent ordering
        user_annotations.sort(key=lambda x: x[0])

        # Create a row for each user's annotations
        for _, annotations in user_annotations:
            user_data = []
            for tag in sorted(all_tags):  # Sort tags for consistent ordering
                # Find if this tag exists in user's annotations
                tag_value = None
                for t, v in annotations:
                    if t == tag:
                        tag_value = v
                        break
                user_data.append(tag_value)
            reliability_data.append(user_data)

    # Convert to numpy array for krippendorff
    reliability_data = np.array(reliability_data)

    import ipdb

    ipdb.set_trace()

    # Calculate alpha
    alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
    return alpha


def main():
    # Load and process data
    data_by_datum = load_and_filter_data("data/wearing_data.ndjson")
    datum_user_annotations, all_tags = prepare_agreement_data(data_by_datum)

    # Calculate agreement
    alpha = calculate_krippendorff_alpha(datum_user_annotations, all_tags)

    # Get unique users and data points
    unique_users = set(user_id for _, user_id in datum_user_annotations.keys())
    unique_data = set(datum_id for datum_id, _ in datum_user_annotations.keys())

    print(f"Krippendorff's alpha: {alpha:.3f}")
    print(f"Number of users: {len(unique_users)}")
    print(f"Number of unique tags: {len(all_tags)}")
    print(f"Number of complete data points: {len(unique_data)}")


if __name__ == "__main__":
    main()
