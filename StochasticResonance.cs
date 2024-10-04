using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class StochasticResonance
{
    public static float Compute(float signal, float noise)
    {
        float snr = signal / noise;
        return snr * noise;
    }
}