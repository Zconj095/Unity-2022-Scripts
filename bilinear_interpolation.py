def bilinear_interpolation(x, x0, x1, y0, y1, x00, y00, x10, y10, x01, y01, x11, y11):
    """Bilinear interpolation at point (x, y) using four corner points.

    Args:
        x (float): The x-coordinate of the point to interpolate.
        x0 (float): The x-coordinate of the top-left corner point.
        x1 (float): The x-coordinate of the top-right corner point.
        y0 (float): The y-coordinate of the top-left corner point.
        y1 (float): The y-coordinate of the top-right corner point.
        x00 (float): The x-coordinate of the bottom-left corner point.
        y00 (float): The y-coordinate of the bottom-left corner point.
        x10 (float): The x-coordinate of the bottom-right corner point.
        y10 (float): The y-coordinate of the bottom-right corner point.
        x01 (float): The x-coordinate of the top-left bottom corner point.
        y01 (float): The y-coordinate of the top-left bottom corner point.
        x11 (float): The x-coordinate of the top-right bottom corner point.
        y11 (float): The y-coordinate of the top-right bottom corner point.

    Returns:
        float: The interpolated value at point (x, y).
    """
    if x <= x0:
        if y <= y0:
            return y00
        elif y <= y1:
            return (y - y0) / (y1 - y0) * (x01 - x00) + y00
        else:
            return y11
    elif x <= x1:
        if y <= y0:
            return (x - x0) / (x1 - x0) * (y01 - y00) + y00
        elif y <= y1:
            return (x - x0) / (x1 - x0) * (y01 - y00) + y00
        else:
            return (x - x0) / (x1 - x0) * (y11 - y10) + y10
    else:
        if y <= y0:
            return (y - y0) / (y1 - y0) * (x11 - x10) + y10
        elif y <= y1:
            return (y - y0) / (y1 - y0) * (x11 - x10) + y10
        else:
            return y11