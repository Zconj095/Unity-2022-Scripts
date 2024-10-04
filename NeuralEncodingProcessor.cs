using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class NeuralEncodingProcessor: MonoBehaviour
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

    public static float[] StatisticalEncoding(float[] differences)
    {
        float[] encodedValues = new float[differences.Length];
        for (int i = 0; i < differences.Length; i++)
        {
            encodedValues[i] = (float)Math.Exp(-Math.Pow(differences[i], 2) / (2 * 1e-6f));
        }
        return encodedValues;
    }

    public static float[] StimulusBasedCoding(float[] stimulus)
    {
        float[] neuralActivities = new float[stimulus.Length];
        for (int i = 0; i < stimulus.Length; i++)
        {
            neuralActivities[i] = (float)Math.Exp(-Math.Pow(stimulus[i], 2) / (2 * 1e-6f));
        }
        return neuralActivities;
    }

    public static float[] PhysicallyBasedEncoding(float[] stimulus)
    {
        // This function is identical to StimulusBasedCoding in behavior.
        return StimulusBasedCoding(stimulus);
    }

    public static float[] RelationallyDeterminedProcess(float[] differences)
    {
        // This function is identical to StatisticalEncoding in behavior.
        return StatisticalEncoding(differences);
    }
}