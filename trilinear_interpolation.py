def trilinear_interpolation(x, y, z, x0, x1, x2, y0, y1, y2, z00, z01, z11, z10, z20):
    """
    Trilinear interpolation at point (x, y, z) using 8 corner points.

    Args:
        x (float): The x-coordinate of the point to interpolate.
        y (float): The y-coordinate of the point to interpolate.
        z (float): The z-coordinate of the point to interpolate.
        x0 (float): The x-coordinate of the bottom-left corner point.
        x1 (float): The x-coordinate of the bottom-right corner point.
        x2 (float): The x-coordinate of the top-left corner point.
        y0 (float): The y-coordinate of the bottom-left corner point.
        y1 (float): The y-coordinate of the bottom-right corner point.
        y2 (float): The y-coordinate of the top-left corner point.
        z00 (float): The value at the bottom-left corner point.
        z01 (float): The value at the bottom-center point.
        z11 (float): The value at the top-right corner point.
        z10 (float): The value at the bottom-right corner point.
        z20 (float): The value at the top-left corner point.

    Returns:
        float: The interpolated value at point (x, y, z).
    """
    if x <= x0:
        if y <= y0:
            if z <= z00:
                return z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            else:
                return z20
        elif y <= y1:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return (z - z10) / (z20 - z10) * (z20 - z10) + z20
        else:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return z20
    elif x <= x1:
        if y <= y0:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return z20
        elif y <= y1:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return (z - z10) / (z20 - z10) * (z20 - z10) + z20
        else:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return z20
    else:
        if y <= y0:
            if z <= z00:
                return z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            else:
                return z20
        elif y <= y1:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return (z - z10) / (z20 - z10) * (z20 - z10) + z20
        else:
            if z <= z00:
                return (z - z00) / (z10 - z00) * (z10 - z00) + z00
            elif z <= z10:
                return (z - z00) / (z10 - z00) * (z11 - z10) + z10
            else:
                return z20