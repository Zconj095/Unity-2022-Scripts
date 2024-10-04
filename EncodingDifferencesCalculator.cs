using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class EncodingDifferencesCalculator: MonoBehaviour
{
    public static float[] EncodingDifferences(float[] patterns)
    {
        float[] differences = new float[patterns.Length - 1];
        for (int i = 0; i < patterns.Length-1; i++)
        {
            differences[i] = patterns[i + 1] - patterns[i];
        }
        return differences;
    }

    public static void Main()
    {
        float[] populationActivity = GenerateLinearSpace(-10f, 10f, 100);
        float[] encodedDifferences = EncodingDifferences(populationActivity);

        Console.WriteLine("Encoded Differences:");
        foreach (float diff in encodedDifferences)
        {
            Console.WriteLine(diff);
        }
    }
    
    public static float[] GenerateLinearSpace(float start, float end, int num)
    {
        float[] result = new float[num];
        float step = (end - start) / (num - 1);
        
        for (int i = 0; i < num; i++)
        {
            result[i] = start + i * step;
        }
        
        return result;
    }
}