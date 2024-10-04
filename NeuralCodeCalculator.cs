using System;
using System.Collections.Generic;
using UnityEngine;
public class NeuralCodeCalculator: MonoBehaviour
{
    public static float NeuralCode(float[] neurons, float[] connections)
    {
        float sum = 0f;
        for (int i = 0; i < neurons.Length; i++)
        {
            float harmonics = (float)Math.Exp(-neurons[i] * neurons[i]) * (float)Math.Sin(connections[i]);
            sum += harmonics;
        }
        return sum;
    }

    public static void Main()
    {
        float[] neurons = GenerateLinearSpace(-10f, 10f, 100);
        float[] connections = GenerateLinearSpace(-10f, 10f, 100);

        float neuralCodeValue = NeuralCode(neurons, connections);
        Console.WriteLine("Neural Code Value: " + neuralCodeValue);
    }

    private static float[] GenerateLinearSpace(float start, float end, int num)
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