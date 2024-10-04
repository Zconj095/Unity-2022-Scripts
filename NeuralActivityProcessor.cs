using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class NeuralActivityProcessor: MonoBehaviour
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

    public static void Main()
    {
        float[] regionalActivity = GenerateLinearSpace(-10f, 10f, 100);
        float[] encodedDifferences = EncodingDifferences(regionalActivity);

        Console.WriteLine("Encoded Differences:");
        foreach (float diff in encodedDifferences)
        {
            Console.WriteLine(diff);
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