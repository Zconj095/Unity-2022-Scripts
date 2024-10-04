using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class NeuralCodeCalculator1: MonoBehaviour
{
    public static float[] NeuralCode(float[] level1, float[] level2, float[] level3)
    {
        float[] code = new float[level1.Length];
        for (int i = 0; i < level1.Length; i++)
        {
            code[i] = (float)(Math.Exp(-level1[i] * level1[i]) * Math.Sin(level2[i]) 
                               + Math.Exp(-level3[i] * level3[i]) * Math.Cos(level3[i]));
        }
        return code;
    }

    public static void Main()
    {
        float[] level1 = GenerateLinearSpace(-10f, 10f, 100);
        float[] level2 = GenerateLinearSpace(-10f, 10f, 100);
        float[] level3 = GenerateLinearSpace(-10f, 10f, 100);

        float[] codeValue = NeuralCode(level1, level2, level3);
        Console.WriteLine("Neural Code Values:");
        foreach (float value in codeValue)
        {
            Console.WriteLine(value);
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