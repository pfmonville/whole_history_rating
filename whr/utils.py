class UnstableRatingException(Exception):
	pass

def test_stability(v1, v2, precision=10e-3):
    """tests if two lists of lists of floats are equal but a certain precision
    
    Args:
        v1 (list[list[float]]): first list containing ints
        v2 (list[list[float]]): second list containing ints
        precision (float, optional): the precision after which v1 and v2 are not equal
    
    Returns:
        bool: True if the two lists are close enought, False otherwise
    """
    v1 = [x for y in v1 for x in y]
    v2 = [x for y in v2 for x in y]
    for x1, x2 in zip(v1, v2):
        if abs(x2 - x1) > precision:
            return False
    return True