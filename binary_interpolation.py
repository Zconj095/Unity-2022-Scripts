def binary_interpolation(x, x0, x1, y0, y1):
    """
    Returns the interpolated value at x using binary interpolation.

    Args:
        x (float): The point at which to interpolate.
        x0 (float): The x-coordinate of the first point.
        x1 (float): The x-coordinate of the second point.
        y0 (float): The y-coordinate of the first point.
        y1 (float): The y-coordinate of the second point.

    Returns:
        float: The interpolated value at x.
    """
    if x <= x0:
        return y0
    elif x >= x1:
        return y1
    else:
        h = (x - x0) / (x1 - x0)
        return y0 + h * (y1 - y0)