using UnityEngine;
using System.Collections.Generic;

public class ConvexHullAreaCalculator : MonoBehaviour {

    public Vector2[] points = new Vector2[] {
        new Vector2(1, 3), new Vector2(3, 4), new Vector2(5, 7), new Vector2(3, 1), new Vector2(6, 7), new Vector2(5, 2)
    };

    // Calculate area of Convex Hull
    public float CalculateArea() {
        List<Vector2> convexHull = FindConvexHull(points);
        return CalculatePolygonArea(convexHull);
    }

    private List<Vector2> FindConvexHull(Vector2[] points) {
        List<Vector2> hull = new List<Vector2>();

        // Find the leftmost point
        Vector2 startPoint = points[0];
        foreach (Vector2 pt in points) {
            if (pt.x < startPoint.x) {
                startPoint = pt;
            }
        }

        Vector2 currentPoint = startPoint;
        Vector2 endPoint;
        do {
            hull.Add(currentPoint);
            endPoint = points[0];

            for (int i = 1; i < points.Length; i++) {
                if (endPoint == currentPoint || IsCounterClockwise(currentPoint, endPoint, points[i])) {
                    endPoint = points[i];
                }
            }
            currentPoint = endPoint;
        }
        while (endPoint != startPoint); // Return to the starting point

        return hull;
    }

    private bool IsCounterClockwise(Vector2 a, Vector2 b, Vector2 c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
    }

    private float CalculatePolygonArea(List<Vector2> polygon) {
        float area = 0;
        int j = polygon.Count - 1;

        for (int i = 0; i < polygon.Count; i++) {
            area += (polygon[j].x + polygon[i].x) * (polygon[j].y - polygon[i].y);
            j = i;  // j is previous vertex to i
        }

        return Mathf.Abs(area / 2.0f);
    }

    // Using OnDrawGizmos to Visualize the Convex Hull
    void OnDrawGizmos() {
        Gizmos.color = Color.red;
        List<Vector2> hull = FindConvexHull(points);
        for (int i = 0; i < hull.Count; i++) {
            Gizmos.DrawSphere(hull[i], 0.1f);
            if (i < hull.Count - 1) {
                Gizmos.DrawLine(hull[i], hull[i + 1]);
            } else {
                Gizmos.DrawLine(hull[i], hull[0]);
            }
        }
    }
}