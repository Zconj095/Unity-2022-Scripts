using System;
using UnityEngine;

public class SimulatedNeuralNetwork : MonoBehaviour
{
    // Network Parameters, assume a simple one-layer network
    private float[] weights; // Represents the strength of connections in the network
    private float bias = 0;  // Additional bias

    void Start()
    {
        InitializeNetwork();
    }

    void InitializeNetwork()
    {
        // Initialize weights randomly or based on some predefined logic
        int inputSize = 3; // Number of inputs
        weights = new float[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = UnityEngine.Random.Range(-1.0f, 1.0f);
        }
        bias = UnityEngine.Random.Range(-1.0f, 1.0f);
    }

    public float[] Predict(float[] inputs)
    {
        float[] outputs = new float[1]; // Configure according to your output needs
        float sum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * weights[i];
        }
        outputs[0] = ActivationFunction(sum + bias);
        return outputs;
    }

    float ActivationFunction(float value)
    {
        // Using a basic sigmoid activation function
        return 1.0f / (1.0f + Mathf.Exp(-value));
    }
}