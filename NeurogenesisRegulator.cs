using UnityEngine;
using System.Collections.Generic;

public class NeurogenesisRegulator : MonoBehaviour
{
    private Dictionary<string, float> levels;

    public NeurogenesisRegulator(string[] factors)
    {
        levels = new Dictionary<string, float>();
        foreach (string factor in factors)
        {
            levels.Add(factor, 0);
        }
    }

    // Updated method name to avoid collision with Unity's lifecycle method
    public void UpdateFactorLevel(string factor, float amount)
    {
        if (levels.ContainsKey(factor))
        {
            levels[factor] += amount;
        }
    }

    public void Propagate()
    {
        foreach (KeyValuePair<string, float> level in levels)
        {
            if (level.Value > 0.1f)
            {
                TriggerNeurogenesis();
                break;
            }
        }
    }

    private void TriggerNeurogenesis()
    {
        Debug.Log("Neurogenesis activated!");
    }
}