using UnityEngine;
using System.Collections.Generic;

public class SimpleGraph : MonoBehaviour {
    private Dictionary<int, List<int>> _adjacencyList = new Dictionary<int, List<int>>();

    void Start() {
        // Example Usage:
        for (int i = 0; i < 10; i++) {
            AddVertex(i);
        }
        AddEdge(0, 1);
        AddEdge(1, 2);
        AddEdge(2, 3);
        AddEdge(4, 5);
        AddEdge(5, 6);
        AddEdge(7, 8);
        AddEdge(8, 9);

        int numCommunities = DetectCommunities();
        Debug.Log("Number of communities: " + numCommunities);
    }

    public void AddVertex(int vertex) {
        _adjacencyList[vertex] = new List<int>();
    }

    public void AddEdge(int vertex1, int vertex2) {
        if (_adjacencyList.ContainsKey(vertex1) && _adjacencyList.ContainsKey(vertex2)) {
            _adjacencyList[vertex1].Add(vertex2);
            _adjacencyList[vertex2].Add(vertex1);
        }
    }

    public int DetectCommunities() {
        bool[] visited = new bool[_adjacencyList.Count];
        int numCommunities = 0;

        // Iterate through all vertices to find unvisited vertex, which marks the start of a new community
        for (int i = 0; i < visited.Length; i++) {
            if (!visited[i]) {
                // Start DFS from this vertex marking new community
                DFS(i, visited);
                numCommunities++;
            }
        }

        return numCommunities;
    }

    private void DFS(int vertex, bool[] visited) {
        Stack<int> stack = new Stack<int>();
        stack.Push(vertex);
        visited[vertex] = true;

        while (stack.Count > 0) {
            int current = stack.Pop();
            foreach (int neighbor in _adjacencyList[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.Push(neighbor);
                }
            }
        }
    }
}