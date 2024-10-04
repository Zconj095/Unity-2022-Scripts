using UnityEngine;
using System.Linq;

public class SynapseFilter : MonoBehaviour
{
    private float[] weights;
    private float threshold = 0.1f;

    public SynapseFilter(float[] initialWeights)
    {
        weights = initialWeights.Select(weight => Mathf.Abs(weight)).ToArray();
    }

    public float[] Sample(int nSamples)
    {
        float totalWeight = weights.Sum();
        float[] probabilities = weights.Select(weight => weight / totalWeight).ToArray();
        float[] sampledWeights = new float[nSamples];

        for (int i = 0; i < nSamples; i++)
        {
            float randomPoint = Random.value * totalWeight;
            float cumulative = 0;

            for (int j = 0; j < weights.Length; j++)
            {
                cumulative += probabilities[j];
                if (randomPoint <= cumulative)
                {
                    sampledWeights[i] = weights[j];
                    break;
                }
            }
        }
        return sampledWeights;
    }
}