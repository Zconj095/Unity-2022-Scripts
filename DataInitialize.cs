using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DataInitialize : MonoBehaviour
{
    void Start()
    {
        // Example data initialization
        float[] stimuli1 = RandomValues(1000);
        float[] stimuli2 = RandomValues(1000);
        float[] differences = NeuralCodingProcessor.DifferenceBasedCoding(stimuli1, stimuli2);
        float[] stimulus = RandomValues(1000);
        float[] neuralActivityValues = NeuralCodingProcessor.NeuralActivity(differences, stimulus);

        foreach (var activity in neuralActivityValues)
        {
            Debug.Log("Neural Activity: " + activity);
        }
    }

    float[] RandomValues(int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = UnityEngine.Random.value;
        return values;
    }
}
