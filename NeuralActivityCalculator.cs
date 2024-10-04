using System;
using UnityEngine;
public class NeuralActivityCalculator: MonoBehaviour
{
    public static float[] CalculateNeuralCode(float[] neuralActivity)
    {
        float[] code = new float[neuralActivity.Length];
        for (int i = 0; i < neuralActivity.Length; i++)
        {
            code[i] = (float)Math.Exp(-neuralActivity[i] * neuralActivity[i]) * (float)Math.Sin(neuralActivity[i]);
        }
        return code;
    }

    public static void Main()
    {
        float[] neuralActivity = GenerateLinearSpace(-10f, 10f, 100);
        float[] codeValue = CalculateNeuralCode(neuralActivity);
        Console.WriteLine("Neural Code Values:");
        foreach (float code in codeValue)
        {
            Console.WriteLine(code);
        }
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