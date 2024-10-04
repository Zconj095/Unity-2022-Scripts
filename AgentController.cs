using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentController : MonoBehaviour
{
    SimulatedNeuralNetwork network;

    void Awake()
    {
        network = GetComponent<SimulatedNeuralNetwork>();
    }

    void Update()
    {
        // Example data feed
        float[] sensorData = new float[] { 0.5f, 0.3f, 0.9f }; // Assume some real sensor data here
        float[] result = network.Predict(sensorData);

        // Use result to control the agent
        Debug.Log("Output from network: " + result[0]);
    }
}