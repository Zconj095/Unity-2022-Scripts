def trinary_interpolation(x, x0, x1, x2, y0, y1, y2):
    """
    Returns the interpolated value at x using trinary interpolation.

    Args:
        x (float): The point at which to interpolate.
        x0 (float): The x-coordinate of the first point.
        x1 (float): The x-coordinate of the second point.
        x2 (float): The x-coordinate of the third point.
        y0 (float): The y-coordinate of the first point.
        y1 (float): The y-coordinate of the second point.
        y2 (float): The y-coordinate of the third point.

    Returns:
        float: The interpolated value at x.
    """
    if x <= x0:
        return y0
    elif x >= x2:
        return y2
    elif x <= (x1 + x2) / 2:
        h = (x - x0) / (x1 - x0)
        return y0 + h * (y1 - y0)
    else:
        k = (x - x1) / (x2 - x1)
        return y1 + k * (y2 - y1)