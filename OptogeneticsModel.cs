using System;
using System.Linq;
using UnityEngine;

public class OptogeneticsModel: MonoBehaviour
{
    private string[] opsins;
    private float[] wavelengths;

    public OptogeneticsModel(string[] ops, float[] wlen)
    {
        opsins = ops;
        wavelengths = wlen;
    }

    public float[] Stimulate(float[] intensities)
    {
        // Instead of 'params', use 'parameters'
        Func<float[], float> CostFunction = (parameters) =>
        {
            float activity = 0f;
            for (int i = 0; i < parameters.Length; i++)
            {
                float sigmoid = 1f / (1f + Mathf.Exp(-parameters[i] * intensities[i]));
                activity += sigmoid * wavelengths[i]; // assuming wavelengths correlate to activity
            }
            return -activity; // since we are minimizing
        };

        float[] parameters = new float[opsins.Length];
        for (int i = 0; i < opsins.Length; i++)
        {
            parameters[i] = 0.5f; // Initial guess for parameters
        }

        float learningRate = 0.01f;
        int maxIterations = 1000;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            float[] gradients = new float[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                float originalParam = parameters[i];

                // Modify parameter slightly to compute gradient
                parameters[i] = originalParam + 0.01f;
                float costPlus = CostFunction(parameters);

                parameters[i] = originalParam - 0.01f;
                float costMinus = CostFunction(parameters);

                // Calculate gradient
                gradients[i] = (costPlus - costMinus) / 0.02f;

                // Reset the parameter to its original value
                parameters[i] = originalParam;
            }

            // Update parameters based on gradients
            for (int i = 0; i < parameters.Length; i++)
            {
                parameters[i] -= learningRate * gradients[i];
            }
        }

        return parameters;
    }
}