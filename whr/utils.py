from __future__ import annotations


class UnstableRatingException(Exception):
    pass


def test_stability(
    v1: list[list[float]], v2: list[list[float]], precision: float = 10e-3
) -> bool:
    """Tests whether two lists of lists of floats are approximately equal within a specified precision.

    This function flattens each list of lists into a single list and compares each corresponding element from the two lists. If the absolute difference between any pair of elements exceeds the given precision, the lists are considered not equal.

    Args:
        v1 (list[list[float]]): The first list of lists of floats.
        v2 (list[list[float]]): The second list of lists of floats.
        precision (float, optional): The precision threshold below which the values are considered equal. Defaults to 0.01.

    Returns:
        bool: True if the two lists are considered close enough, i.e., no pair of corresponding elements differs by more than the specified precision. False otherwise.
    """
    v1_flattened = [x for y in v1 for x in y]
    v2_flattened = [x for y in v2 for x in y]
    for x1, x2 in zip(v1_flattened, v2_flattened):
        if abs(x2 - x1) > precision:
            return False
    return True
