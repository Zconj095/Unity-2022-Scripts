using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;

public class NeuroscienceComputations: MonoBehaviour
{
    public static float[] LowFrequencyFluctuations(float[] time, float frequency)
    {
        float[] output = new float[time.Length];
        for (int i = 0; i < time.Length; i++)
        {
            output[i] = (float)Math.Sin(2 * Math.PI * frequency * time[i]);
        }
        return output;
    }

    public static float[] NeuralActivity(float[] time, float gain)
    {
        float[] output = new float[time.Length];
        for (int i = 0; i < time.Length; i++)
        {
            output[i] = (float)Math.Exp(-Math.Pow(time[i], 2) / (2 * gain * gain));
        }
        return output;
    }

    public static float TemporalIntegration(float[] time, float[] functionValues)
    {
        float integral = 0f;
        float dt = time[1] - time[0];
        for (int i = 0; i < functionValues.Length - 1; i++)
        {
            integral += (functionValues[i] + functionValues[i + 1]) / 2 * dt;
        }
        return integral;
    }
  
    public static float FunctionalConnectivity(float[] activity1, float[] activity2)
    {
        // Simplified correlation calculation
        float mean1 = 0, mean2 = 0, denom1 = 0, denom2 = 0, numer = 0;
        for (int i = 0; i < activity1.Length; i++)
        {
            mean1 += activity1[i];
            mean2 += activity2[i];
        }
        mean1 /= activity1.Length;
        mean2 /= activity2.Length;

        for (int i = 0; i < activity1.Length; i++)
        {
            numer += (activity1[i] - mean1) * (activity2[i] - mean2);
            denom1 += (activity1[i] - mean1) * (activity1[i] - mean1);
            denom2 += (activity2[i] - mean2) * (activity2[i] - mean2);
        }
        return numer / (float)Math.Sqrt(denom1 * denom2);
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