using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NoisyChannel: MonoBehaviour
{
    public float Generate(float t, float fc, float dc)
    {
        // Simplified noise generation
        return 0.7f * Mathf.Abs(Mathf.Sin(fc * t)) + dc * UnityEngine.Random.value;
    }
}