using UnityEngine;

public class Neurogenesis : MonoBehaviour
{
    public int progenitors;
    public float differentiationRate;
    public float apoptosisRate;

    public int SimulateNeurogenesis()
    {
        int nNeurons = 0;
        int nProgenitors = progenitors;

        for (int day = 0; day < 365; day++)
        {
            int nDifferentiated = Mathf.FloorToInt(differentiationRate * nProgenitors);
            int nApoptosis = Mathf.FloorToInt(apoptosisRate * nNeurons);

            nProgenitors -= nDifferentiated;
            nNeurons += nDifferentiated - nApoptosis;
        }

        return nNeurons;
    }
}