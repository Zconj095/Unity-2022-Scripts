using System.Collections;
using System.Collections.Generic;
using UnityEngine;
class DataInitialize1 : MonoBehaviour
{
    void Start()
    {
        // Example data initialization
        float[] stimuli1 = GenerateRandomData(1000);
        float[] stimuli2 = GenerateRandomData(1000);

        // Processing data
        float[] differences = NeuralEncodingProcessor.DifferenceBasedCoding(stimuli1, stimuli2);
        float[] statisticallyEncoded = NeuralEncodingProcessor.StatisticalEncoding(differences);
        float[] relationallyDetermined = NeuralEncodingProcessor.RelationallyDeterminedProcess(differences);

        float[] stimulus = GenerateRandomData(1000);
        float[] physicallyEncoded = NeuralEncodingProcessor.PhysicallyBasedEncoding(stimulus);

        // Logging the output
        Debug.Log("Statistically encoded values: " + ArrayToString(statisticallyEncoded));
        Debug.Log("Physically encoded values: " + ArrayToString(physicallyEncoded));
        Debug.Log("Relationally determined values: " + ArrayToString(relationallyDetermined));
    }

    float[] GenerateRandomData(int length)
    {
        float[] data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = UnityEngine.Random.value;
        }
        return data;
    }

    string ArrayToString(float[] array)
    {
        return "[" + string.Join(", ", array) + "]";
    }
}