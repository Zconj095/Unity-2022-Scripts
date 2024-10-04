using System;
using System.Linq;
using UnityEngine;
public class NeuralProcessing: MonoBehaviour
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

    public static Tuple<float[], int[]> RelationalDetermination(float[] stimuli, int bins)
    {
        float min = stimuli.Min();
        float max = stimuli.Max();
        float range = max - min;
        float binSize = range / bins;
        float[] binLimits = new float[bins + 1];
        int[] binCounts = new int[bins];

        for (int i = 0; i <= bins; i++)
        {
            binLimits[i] = min + i * binSize;
        }

        foreach (float val in stimuli)
        {
            int binIndex = (int)((val - min) / binSize);
            if (binIndex == bins) binIndex--;
            binCounts[binIndex]++;
        }
        return new Tuple<float[], int[]>(binLimits, binCounts);
    }

    public static void Main()
    {
        float[] neuralActivity = GenerateLinearSpace(-10f, 10f, 100);
        float[] encodedDiff = DifferenceBasedCoding(neuralActivity);
        Tuple<float[], int[]> relationalDetermination = RelationalDetermination(neuralActivity, 10);

        Console.WriteLine("Encoded Differences:");
        foreach (float diff in encodedDiff)
        {
            Console.WriteLine(diff);
        }
        
        Console.WriteLine("Relational Determination (Hist):");
        for (int i = 0; i < relationalDetermination.Item2.Length; i++)
        {
            Console.WriteLine($"Bin {i}: {relationalDetermination.Item2[i]}");
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