using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class NeuralActivityProcessor2: MonoBehaviour
{
    public static float[] DifferenceBasedCoding(float[] stimuli1, float[] stimuli2)
    {
        if (stimuli1.Length != stimuli2.Length)
            throw new ArgumentException("Stimuli arrays must have the same length.");

        float[] differences = new float[stimuli1.Length];
        for (int i = 0; i < stimuli1.Length; i++)
        {
            differences[i] = Math.Abs(stimuli1[i] - stimuli2[i]);
        }
        return differences;
    }

    public static float SpatialDifference(float[] stimuli1, float[] stimuli2)
    {
        if (stimuli1.Length != stimuli2.Length)
            throw new ArgumentException("Stimuli arrays must have the same length.");

        float sumOfSquares = 0;
        for (int i = 0; i < stimuli1.Length; i++)
        {
            sumOfSquares += (stimuli1[i] - stimuli2[i]) * (stimuli1[i] - stimuli2[i]);
        }
        return (float)Math.Sqrt(sumOfSquares);
    }

    public static float[] TemporalDifference(float[] stimuli1, float[] stimuli2)
    {
        if (stimuli1.Length != stimuli2.Length)
            throw new ArgumentException("Stimuli arrays must have the same length.");

        float[] diffStimuli1 = new float[stimuli1.Length - 1];
        float[] diffStimuli2 = new float[stimuli2.Length - 1];
        
        for (int i = 1; i < stimuli1.Length; i++)
        {
            diffStimuli1[i - 1] = stimuli1[i] - stimuli1[i - 1];
            diffStimuli2[i - 1] = stimuli2[i] - stimuli2[i - 1];
        }

        float[] temporalDifferences = new float[diffStimuli1.Length];
        for (int i = 0; i < diffStimuli1.Length; i++)
        {
            temporalDifferences[i] = Math.Abs(diffStimuli1[i] - diffStimuli2[i]);
        }
        return temporalDifferences;
    }

    public static float[] NeuralActivity(float[] differences)
    {
        // Assuming use of a very small constant for scale in neural response
        float[] neuralActivityValues = new float[differences.Length];
        for (int i = 0; i < differences.Length; i++)
        {
            neuralActivityValues[i] = (float)Math.Exp(-Math.Pow(differences[i], 2) / (2 * 1e-6f));
        }
        return neuralActivityValues;
    }
}