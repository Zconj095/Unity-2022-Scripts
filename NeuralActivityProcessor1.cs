using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class NeuralActivityProcessor1 : MonoBehaviour
{
    public static float[] DifferenceBasedCoding(float[] stimuli)
    {
        float[] differences = new float[stimuli.Length - 1];
        for (int i = 0; i < stimuli.Length - 1; i++)
        {
            differences[i] = stimuli[i + 1] - stimuli[i];
        }
        return differences;
    }

    public static float[,] StimulusBasedCoding(float[] stimuli)
    {
        float[,] output = new float[1, stimuli.Length];
        for (int i = 0; i < stimuli.Length; i++)
        {
            output[0, i] = stimuli[i];
        }
        return output;
    }

    public static void Main()
    {
        float[] neuralActivity = GenerateLinearSpace(-10f, 10f, 100);
        float[] encodedDifferences = DifferenceBasedCoding(neuralActivity);
        float[,] encodedStimuli = StimulusBasedCoding(neuralActivity);

        Console.WriteLine("Encoded Differences:");
        foreach (float diff in encodedDifferences)
        {
            Console.WriteLine(diff);
        }

        Console.WriteLine("Encoded Single Stimuli:");
        for (int i = 0; i < neuralActivity.Length; i++)
        {
            Console.Write(encodedStimuli[0, i] + " ");
        }
        Console.WriteLine();
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