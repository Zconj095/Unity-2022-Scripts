using UnityEngine;
using System;
public class Neuron : MonoBehaviour
{
    public float voltage = -70.0f;
    
    public void UpdateVoltage(float current)
    {
        voltage += 0.5f * current;
    }
}