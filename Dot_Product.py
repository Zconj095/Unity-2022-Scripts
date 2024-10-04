def dot_product(vector1, vector2):
    """
    Returns the dot product of two vectors.

    Args:
        vector1 (Vector3): The first vector.
        vector2 (Vector3): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

