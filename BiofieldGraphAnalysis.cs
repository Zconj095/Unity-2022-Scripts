using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class BiofieldGraphAnalysis : MonoBehaviour {
    private Dictionary<int, List<int>> graph;
    private int numVertices = 50;

    void Start() {
        BuildGraph(numVertices);
        int diameter = CalculateDiameter();
        Debug.Log($"Graph Diameter: {diameter}");
    }

    void BuildGraph(int numVertices) {
        graph = new Dictionary<int, List<int>>();

        // Initialize graph
        for (int i = 0; i < numVertices; i++) {
            graph[i] = new List<int>();
        }

        // Add edges randomly
        for (int i = 0; i < numVertices; i++) {
            for (int j = i + 1; j < numVertices; j++) {
                if (Random.value < 0.3f) {
                    graph[i].Add(j);
                    graph[j].Add(i); // Since it's an undirected graph
                }
            }
        }
    }

    private int CalculateDiameter() {
        int maxDiameter = 0;

        // Simple algorithm to approximate the diameter of a graph
        foreach (var source in graph.Keys) {
            var dist = BreadthFirstSearch(source);
            int currentMax = dist.Max();
            if (currentMax > maxDiameter)
                maxDiameter = currentMax;
        }
        return maxDiameter;
    }

    private int[] BreadthFirstSearch(int source) {
        Queue<int> queue = new Queue<int>();
        int[] distances = new int[numVertices];
        for (int i = 0; i < numVertices; i++) distances[i] = int.MaxValue; // Infinite

        queue.Enqueue(source);
        distances[source] = 0;

        while (queue.Count > 0) {
            int current = queue.Dequeue();

            foreach (var neighbor in graph[current]) {
                if (distances[neighbor] == int.MaxValue) {
                    queue.Enqueue(neighbor);
                    distances[neighbor] = distances[current] + 1;
                }
            }
        }
        return distances;
    }
}