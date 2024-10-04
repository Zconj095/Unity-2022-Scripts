using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class StatisticalFrequencyDistributionCalculator: MonoBehaviour
{
    public static Tuple<float[], int[]> StatisticalFrequencyDistribution(float[] stimuli, int bins)
    {
        float min = float.MaxValue;
        float max = float.MinValue;
        foreach (float val in stimuli)
        {
            if (val < min) min = val;
            if (val > max) max = val;
        }

        float range = max - min;
        float binSize = range / bins;
        float[] binLimits = new float[bins + 1];
        int[] binCounts = new int[bins];
        
        for (int i = 0; i <= bins; i++)
        {
            binLimits[i] = min + i * binSize;
        }

        foreach (float val in stimuli)
        {
            int binIndex = (int)((val - min) / binSize);
            if (binIndex == bins) binIndex--; // Place the upper boundary value in the last bin
            binCounts[binIndex]++;
        }

        return new Tuple<float[], int[]>(binLimits, binCounts);
    }

    public static void Main()
    {
        float[] neuralActivity = GenerateLinearSpace(-10f, 10f, 100);
        Tuple<float[], int[]> frequencyDistribution = StatisticalFrequencyDistribution(neuralActivity, 10);

        Console.WriteLine("Frequency Distribution Bins:");
        for (int i = 0; i < frequencyDistribution.Item2.Length; i++)
        {
            Console.WriteLine("Bin {0}: {1}", i, frequencyDistribution.Item2[i]);
        }
    }
    
    private static float[] GenerateLinearSpace(float start, float end, int count)
    {
        float[] result = new float[count];
        float step = (end - start) / (count - 1);
        for (int i = 0; i < count; i++)
        {
            result[i] = start + step * i;
        }
        return result;
    }
}