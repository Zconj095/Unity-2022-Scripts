using System;
using UnityEngine;

public class ExperienceOptimizer : MonoBehaviour
{
    public float[] parameters;
    private float learningRate = 0.05f;

    // Constructor to initialize the parameters
    public void Initialize(int paramCount)
    {
        parameters = new float[paramCount];
        for (int i = 0; i < paramCount; i++)
        {
            parameters[i] = UnityEngine.Random.value / 2f; // Initialize parameters with random values
        }
    }

    // Function to optimize parameters based on supplied gradients
    public void OptimizeParameters(float[] grads)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            // Applying gradient update with learning rate
            parameters[i] += -learningRate * grads[i];
        }
    }

    // Property to calculate and return the loss
    public float Loss
    {
        get
        {
            float sum = 0f;
            foreach (float param in parameters)
            {
                sum += param;
            }
            return -Mathf.Pow(sum, 2); // Squared sum negative as the loss
        }
    }

    // Example Unity Start method for initialization
    void Start()
    {
        // Initialize with a sample parameter count, e.g., 3
        Initialize(3);

        // Example gradients
        float[] exampleGrads = new float[] {-0.1f, 0.05f, 0.03f};
        OptimizeParameters(exampleGrads);
    }
}