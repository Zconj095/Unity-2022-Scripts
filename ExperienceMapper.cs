using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class ExperienceMapper : Agent
{
    public int numDimensions = 3; // Assume we have measurements in three directions
    private float[] encodedFeatures = new float[3];

    public override void Initialize()
    {
        encodedFeatures = new float[numDimensions];
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Assuming that distances are stored in encodedFeatures updated elsewhere in code
        foreach (var feature in encodedFeatures)
        {
            // Normalize observations to be in range [0, 1]
            sensor.AddObservation(Mathf.Clamp01(feature / 100.0f)); // Assuming max sensor range is 100 units
        }
    }

    // Simulated update function that would be connected to actual sensor data in a real scenario
    public void UpdateSensorData(float[] sensorReadings)
    {
        for (int i = 0; i < numDimensions && i < sensorReadings.Length; i++)
        {
            encodedFeatures[i] = sensorReadings[i];
        }
    }
}