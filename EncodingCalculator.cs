using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class EncodingCalculator: MonoBehaviour
{
    public static float[] NarrowEncoding(float[] stimuli)
    {
        float[] neuralActivity = new float[stimuli.Length];
        for (int i = 0; i < stimuli.Length; i++)
        {
            neuralActivity[i] = (float)(Math.Exp(-stimuli[i] * stimuli[i]) * Math.Sin(stimuli[i]));
        }
        return neuralActivity;
    }

    public static float[] WideEncoding(float[] stimuli, float[] body, float[] brain)
    {
        float[] neuralActivity = new float[stimuli.Length];
        for (int i = 0; i < stimuli.Length; i++)
        {
            neuralActivity[i] = (float)(Math.Exp(-stimuli[i] * stimuli[i]) * Math.Sin(stimuli[i]) +
                                         Math.Exp(-body[i] * body[i]) * Math.Cos(body[i]) +
                                         Math.Exp(-brain[i] * brain[i]) * Math.Tan(brain[i]));
        }
        return neuralActivity;
    }

    public static void Main()
    {
        float[] stimuli = GenerateLinearSpace(-10f, 10f, 100);
        float[] body = GenerateLinearSpace(-10f, 10f, 100);
        float[] brain = GenerateLinearSpace(-10f, 10f, 100);

        float[] narrowActivity = NarrowEncoding(stimuli);
        float[] wideActivity = WideEncoding(stimuli, body, brain);

        Console.WriteLine("Wide Encoded Neural Activities:");
        foreach (float activity in wideActivity)
        {
            Console.WriteLine(activity);
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