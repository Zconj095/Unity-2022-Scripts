using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DataInitialize2 : MonoBehaviour
{
    void Start() {
        float[] x = { 1f, 2f };
        float[] y = NeuralCoding.F(x);
        float[] recoveredX = NeuralCoding.F_inv(y);

        Debug.Log("Recovered x: " + ArrayToString(recoveredX));

        float[] stimuli = { 1f, 2f };
        float[] bodyState = { 0.5f, 1f };
        float[] brainActivity = { 0.3f, 0.7f };
        float[] neuralActivityWide = NeuralCoding.EncodingWide(stimuli, bodyState, brainActivity);

        Debug.Log("Wide Neural Activity: " + ArrayToString(neuralActivityWide));

        float[] currentState = { 2f };
        float[] previousState = { 1.5f };
        float[] error = NeuralCoding.PredictError(currentState, previousState);

        Debug.Log("Membrane Potential Error: " + ArrayToString(error));
    }

    // Utility method to convert array to string for logging
    string ArrayToString(float[] array) {
        return "[" + string.Join(", ", array) + "]";
    }
}
