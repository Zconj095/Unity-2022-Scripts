using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class SimpleClustering : MonoBehaviour {
    public int k = 3;  // Number of clusters
    public Vector2[] dataPoints;
    public Vector2[] centroids;
    private Dictionary<int, List<Vector2>> clusterAssignments;  // Declare it here to expand scope

    void Start() {
        dataPoints = GenerateRandomData(100);
        centroids = InitializeCentroids(k);
        clusterAssignments = new Dictionary<int, List<Vector2>>();  // Initialize it here

        // Initialize clusters
        for (int i = 0; i < k; i++) {
            clusterAssignments[i] = new List<Vector2>();
        }

        // Repeat the clustering process a fixed number of times
        // In practice, you might use a convergence criterion instead
        for (int i = 0; i < 10; i++) {
            AssignClusters();
            RecalculateCentroids();
        }

        Debug.Log("Clustering performed.");
    }

    Vector2[] GenerateRandomData(int count) {
        Vector2[] data = new Vector2[count];
        for (int i = 0; i < count; i++) {
            data[i] = new Vector2(Random.Range(0f, 1f), Random.Range(0f, 1f));
        }
        return data;
    }

    Vector2[] InitializeCentroids(int k) {
        Vector2[] centers = new Vector2[k];
        for (int i = 0; i < k; i++) {
            centers[i] = new Vector2(Random.Range(0f, 1f), Random.Range(0f, 1f));
        }
        return centers;
    }

    void AssignClusters() {
        // Assign each data point to the nearest centroid
        foreach (Vector2 point in dataPoints) {
            int closestCentroid = FindClosestCentroid(point);
            clusterAssignments[closestCentroid].Add(point);
        }
    }

    void RecalculateCentroids() {
        for (int i = 0; i < centroids.Length; i++) {
            if (clusterAssignments[i].Count > 0) {
                // Calculate the new centroid
                float newX = clusterAssignments[i].Average(point => point.x);
                float newY = clusterAssignments[i].Average(point => point.y);
                centroids[i] = new Vector2(newX, newY);
            }
        }
    }

    int FindClosestCentroid(Vector2 point) {
        int closestIndex = 0;
        float minDistance = float.MaxValue;

        for (int i = 0; i < centroids.Length; i++) {
            float distance = Vector2.Distance(point, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closestIndex = i;
            }
        }

        return closestIndex;
    }
}