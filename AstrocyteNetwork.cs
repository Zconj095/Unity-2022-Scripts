using System.Collections.Generic;
using UnityEngine;

public class AstrocyteNetwork : MonoBehaviour
{
    private int numAstrocytes = 100;
    private float pConnect = 0.1f;
    private List<List<int>> graph;

    // Initialization
    void Start()
    {
        GenerateGraph();
    }

    void GenerateGraph()
    {
        graph = new List<List<int>>();
        for (int i = 0; i < numAstrocytes; i++)
        {
            graph.Add(new List<int>());
            for (int j = 0; j < numAstrocytes; j++)
            {
                if (i != j && Random.value < pConnect)
                {
                    graph[i].Add(j);
                }
            }
        }
    }

    public float[][] Simulate(int timesteps)
    {
        float[][] states = new float[timesteps][];
        for (int t = 0; t < timesteps; t++)
        {
            states[t] = new float[numAstrocytes];
            for (int i = 0; i < numAstrocytes; i++)
            {
                states[t][i] = Random.value;
            }
        }
        return states;
    }
}