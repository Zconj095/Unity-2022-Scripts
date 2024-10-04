using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class EncodingDecodingCalculator: MonoBehaviour
{
    public static float[] Encode(float[] stimuli)
    {
        float[] neuralActivity = new float[stimuli.Length];
        for (int i = 0; i < stimuli.Length; i++)
        {
            neuralActivity[i] = (float)(Math.Exp(-stimuli[i] * stimuli[i]) * Math.Sin(stimuli[i]));
        }
        return neuralActivity;
    }

    public static float[] Decode(float[] neuralActivity)
    {
        float[] information = new float[neuralActivity.Length];
        for (int i = 0; i < neuralActivity.Length; i++)
        {
            information[i] = (float)(Math.Exp(-neuralActivity[i] * neuralActivity[i]) * Math.Cos(neuralActivity[i]));
        }
        return information;
    }

    public static void Main()
    {
        float[] stimuli = GenerateLinearSpace(-10f, 10f, 100);
        float[] encodedActivity = Encode(stimuli);
        float[] decodedInfo = Decode(encodedActivity);

        Console.WriteLine("Decoded Information:");
        foreach (float info in decodedInfo)
        {
            Console.WriteLine(info);
        }
    }

    public static float[] GenerateLinearSpace(float start, float end, int count)
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