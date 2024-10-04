def convex_hull(points):
    """
    Returns the convex hull of the given points.

    Args:
        points (list of Vector3): The points to compute the convex hull of.

    Returns:
        list of Vector3: The points of the convex hull.
    """
    def orientation(p, q, r):
        """
        Returns the orientation of three points.

        Args:
            p (Vector3): The first point.
            q (Vector3): The second point.
            r (Vector3): The third point.

        Returns:
            int: The orientation of the three points. 0 if the points are collinear, 1 if the points are clockwise, 2 if the points are counterclockwise.
        """
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise

    def find_hull(points):
        """
        Finds the convex hull of the given points.

        Args:
            points (list of Vector3): The points to compute the convex hull of.

        Returns:
            list of Vector3: The points of the convex hull.
        """
        n = len(points)
        if n < 3:
            return points

        hull = []
        p = 0
        for i in range(1, n):
            if points[i].x < points[p].x:
                p = i

        hull.append(points[p])
        p = (p - 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[(p + 1) % n]) == 2:
                hull.append(points[i])
                p = i
        hull.append(points[p])
        return hull

    return find_hull(points)