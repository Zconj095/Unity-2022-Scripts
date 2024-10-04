using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class BrainActivityProcessor: MonoBehaviour
{
    public static float[] EncodingDifferences(float[] activity)
    {
        float[] differences = new float[activity.Length - 1];
        for (int i = 0; i < activity.Length - 1; i++)
        {
            differences[i] = activity[i + 1] - activity[i];
        }
        return differences;
    }

    public static float[] LowFrequencyFluctuations()
    {
        float[] fluctuations = new float[100];
        for (int i = 0; i < 100; i++)
        {
            fluctuations[i] = (float)Math.Sin(i * 2 * Math.PI / 99);
        }
        return fluctuations;
    }

    // A simple method to mimic functional connectivity.
    // Note: This is a basic placeholder and does not represent true statistical correlation.
    public static float[,] FunctionalConnectivity(float[] activity)
    {
        int len = activity.Length;
        float[,] connectivity = new float[len, len];

        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                connectivity[i, j] = (activity[i] + activity[j]) / 2; // Simplified example
            }
        }
        return connectivity;
    }

    public static void Main()
    {
        float[] neuralActivity = GenerateLinearSpace(-10f, 10f, 100);
        float[] encodedDiff = EncodingDifferences(neuralActivity);
        float[] lowFluctuations = LowFrequencyFluctuations();
        float[,] functionalConnectivity = FunctionalConnectivity(neuralActivity);

        Console.WriteLine("Encoded Differences:");
        foreach (float diff in encodedDiff)
        {
            Console.WriteLine(diff);
        }

        Console.WriteLine("Low Frequency Fluctuations:");
        foreach (float fluc in lowFluctuations)
        {
            Console.WriteLine(fluc);
        }

        Console.WriteLine("Functional Connectivity Matrix:");
        for (int i = 0; i < neuralActivity.Length; i++)
        {
            for (int j = 0; j < neuralActivity.Length; j++)
            {
                Console.Write(functionalConnectivity[i, j] + " ");
            }
            Console.WriteLine();
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