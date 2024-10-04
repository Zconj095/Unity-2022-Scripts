using System;
using System.Collections.Generic;
using UnityEngine;

public class BrainWaves: MonoBehaviour
{
    private float[] ts;
    private Dictionary<string, float[]> waves;

    public BrainWaves()
    {
        ts = new float[1000];
        for (int i = 0; i < ts.Length; i++)
        {
            ts[i] = i * 10.0f / 999;
        }
        
        waves = new Dictionary<string, float[]> {
            { "delta", MakeWave(2f, 1.5f) },
            { "theta", MakeWave(5f, 2f) },
            { "alpha", MakeWave(11f, 2.5f) }
        };
    }

    private float[] MakeWave(float freq, float amplitude)
    {
        float[] result = new float[ts.Length];
        for (int i = 0; i < ts.Length; i++)
        {
            result[i] = amplitude * Mathf.Sin(freq * ts[i]);
        }
        return result;
    }
}