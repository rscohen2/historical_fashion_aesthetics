"""
Helper functions for text span processing given start and end indices.

All functions expect spans to be in the format of ([...start_indices], [...end_indices])
All functions expect these lists of spans to be sorted by start indices.
"""

import numpy as np


def get_closest_span(target_span, source_spans):
    """
    For a given target span, find the closest source span.

    Args:
        target_span (tuple): A tuple of two integers, start and end indices of the target span.
        source_spans (tuple): A tuple of two lists, start and end indices of source spans.

    Returns:
        int: The index of the closest source span to the target span.
    """
    source_starts, source_ends = source_spans

    # Calculate distances from the target span to each source span
    distances = [
        distance(target_span, (source_start, source_end))
        for source_start, source_end in zip(source_starts, source_ends)
    ]

    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    return closest_index


def get_k_closest_spans(target_span, source_spans, k=1):
    """
    For a given target span, find the k closest source spans.

    Args:
        target_span (tuple): A tuple of two integers, start and end indices of the target span.
        source_spans (tuple): A tuple of two lists, start and end indices of source spans.
        k (int): The number of closest spans to return (default is 1).

    Returns:
        list[int]: A list of indices of the k closest source spans to the target span.
    """
    source_starts, source_ends = source_spans

    # Calculate distances from the target span to each source span
    distances = [
        distance(target_span, (source_start, source_end))
        for source_start, source_end in zip(source_starts, source_ends)
    ]

    # Get the indices of the k smallest distances
    closest_indices = np.argsort(distances)[:k]
    return closest_indices.tolist(), [distances[i] for i in closest_indices]


def distance(target_span, source_span):
    """
    Calculate the distance between a target span and a source span.

    Args:
        target_span (tuple): A tuple of two integers, start and end indices of the target span.
        source_span (tuple): A tuple of two integers, start and end indices of the source span.

    Returns:
        int: The distance between the target span and the source span.
    """
    target_start, target_end = target_span
    source_start, source_end = source_span

    # Calculate the distance as the minimum of the absolute differences
    return min(abs(target_start - source_start), abs(target_end - source_end))


def get_overlap_rows_single(target_span, source_spans, partial=False):
    target_start, target_end = target_span
    source_starts, source_ends = source_spans

    # if partial is False, we only consider full overlaps
    # this means when the target_start <= source_start _and_ target_end >= source_end
    # if partial is True, we consider any overlap
    # this means when target_start < source_end _and_ target_end > source_start
    # TODO: we can use numpy searchsorted for efficient searching

    if not partial:
        overlap_indices = []
        for i, (source_start, source_end) in enumerate(zip(source_starts, source_ends)):
            if target_start <= source_start and target_end >= source_end:
                overlap_indices.append(i)
    else:
        # For partial overlaps, we check if any part of the target span overlaps with source spans
        overlap_indices = []
        for i, (source_start, source_end) in enumerate(zip(source_starts, source_ends)):
            if target_start < source_end and target_end > source_start:
                overlap_indices.append(i)
    return overlap_indices


def get_overlap_rows(target_spans, source_spans, partial=False):
    """
    For each target span, find overlapping spans in the source spans.

    Args:
        target_spans (tuple): A tuple of two lists, start and end indices of target spans.
        source_spans (tuple): A tuple of two lists, start and end indices of source spans.
        partial (bool): If True, allows partial overlaps (default is False, meaning only full overlaps are considered).

    Returns:
        list[list[int]]: output[i] is a list of indices of source spans that overlap with target span i.
    """
    target_starts, target_ends = target_spans
    source_starts, source_ends = source_spans

    overlaps = []

    for target_span in zip(target_starts, target_ends):
        overlaps.append(get_overlap_rows_single(target_span, source_spans, partial))

    return overlaps


if __name__ == "__main__":
    # Example usage
    target_spans = list(zip(*[(2, 5)]))
    source_spans = list(zip(*[(0, 1), (1, 3), (2, 4), (4, 6), (4, 5), (5, 7)]))
    print(target_spans)
    print(source_spans)

    print(get_overlap_rows(target_spans, source_spans, partial=False))
    print(get_overlap_rows(target_spans, source_spans, partial=True))
    print(get_closest_span((2, 5), source_spans))
    print(distance((2, 5), (2, 4)))
    print(distance((2, 5), (5, 7)))
