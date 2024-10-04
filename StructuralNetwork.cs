using UnityEngine;

public class StructuralNetwork : MonoBehaviour
{
    private float[,] sc;

    public StructuralNetwork(int numRegions)
    {
        sc = new float[numRegions, numRegions];
        for (int i = 0; i < numRegions; i++)
        {
            for (int j = 0; j < numRegions; j++)
            {
                sc[i, j] = Random.Range(-1f, 1f);
            }
        }
    }

    public float[] ReinforcePath(int[] path, float weightIncrease = 0.5f)
    {
        float[] weights = new float[path.Length - 1];
        for (int i = 1; i < path.Length; i++)
        {
            int from = path[i - 1];
            int to = path[i];
            float weight = sc[from, to];
            sc[from, to] += weightIncrease;
            weights[i - 1] = weight;
        }
        return weights;
    }
}