using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class NeuralCodingProcessor: MonoBehaviour
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

    public static float[] StimulusBasedCoding(float[] stimulus)
    {
        float[] neuralActivities = new float[stimulus.Length];
        for (int i = 0; i < stimulus.Length; i++)
        {
            neuralActivities[i] = (float)Math.Exp(-Math.Pow(stimulus[i], 2) / (2 * 1e-6f));
        }
        return neuralActivities;
    }

    public static float[] BalanceDifferenceStimulus(float[] differences, float[] stimulus)
    {
        if (differences.Length != stimulus.Length)
            throw new ArgumentException("Differences and stimulus arrays must have the same length.");

        float[] balances = new float[differences.Length];
        for (int i = 0; i < differences.Length; i++)
        {
            balances[i] = (float)(Math.Exp(-Math.Pow(differences[i], 2) / (2 * 1e-6f)) + 
                                   Math.Exp(-Math.Pow(stimulus[i], 2) / (2 * 1e-6f)));
        }
        return balances;
    }
    
    public static float[] NeuralActivity(float[] differences, float[] stimulus)
    {
        return BalanceDifferenceStimulus(differences, stimulus); // This is identical to the previous method
    }
}

