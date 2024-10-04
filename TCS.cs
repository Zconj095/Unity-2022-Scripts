using System;
using UnityEngine;

public class TCS : MonoBehaviour
{
    public float power = 1f;

    public float[] Stimulate(float[] intervals)
    {
        float[] results = new float[intervals.Length];
        for (int i = 0; i < intervals.Length; i++)
        {
            results[i] = Mathf.Sin(2 * Mathf.PI * intervals[i]); // tACS waveform
        }
        return results;
    }
}