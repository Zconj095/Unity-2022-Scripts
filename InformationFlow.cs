using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// Equivalent to InformationFlow
public class InformationFlow: MonoBehaviour
{
    private Dictionary<string, List<string>> layersGraph;

    public InformationFlow(Dictionary<string, List<string>> layers)
    {
        layersGraph = new Dictionary<string, List<string>>();
        AddLayers(layers);
    }

    private void AddLayers(Dictionary<string, List<string>> layers)
    {
        foreach (var layer in layers)
        {
            if (!layersGraph.ContainsKey(layer.Key))
            {
                layersGraph.Add(layer.Key, new List<string>());
            }
            foreach (string connection in layer.Value)
            {
                layersGraph[layer.Key].Add(connection);
            }
        }
    }
}