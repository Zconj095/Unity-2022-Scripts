using UnityEngine;
using System.Collections.Generic;

public class QuantumElementThree: MonoBehaviour
{
    public Vector3 position;  // Spatial coordinate in 3D space
    public float energyLevel; // Energy associated with this quantum element
    public Complex probabilityWaveFunction; // Quantum state of the element
    public Vector4 hyperDimensionalCoordinate; // Represents extra-dimensional influence
    
    public ElementalTypeTwo elementalAttribute;  // Elemental type (e.g., Fire, Water)
    public float elementalIntensity;          // Intensity of the elemental power

    // New: List of possible connections to other elements
    public List<QuantumElementConnection> possibleConnections;

    // Constructor
    public QuantumElementThree(Vector3 pos, float energy, Complex waveFunc, Vector4 hyperDimCoord, ElementalTypeTwo element, float intensity)
    {
        position = pos;
        energyLevel = energy;
        probabilityWaveFunction = waveFunc;
        hyperDimensionalCoordinate = hyperDimCoord;
        elementalAttribute = element;
        elementalIntensity = intensity;
        possibleConnections = new List<QuantumElementConnection>();
    }

    // Add a possible connection to another element in the network
    public void AddConnection(QuantumElementThree targetElement, float probability)
    {
        QuantumElementConnection connection = new QuantumElementConnection(targetElement, probability);
        possibleConnections.Add(connection);
    }


    // Fix: TransitionState method is defined to transition the element's quantum state.
    public void TransitionState(Complex newWaveFunc, float timeDelta)
    {
        probabilityWaveFunction = Complex.Lerp(probabilityWaveFunction, newWaveFunc, timeDelta);
    }

    // Transition based on probability and the possibility network
    public void TransitionBasedOnNetwork(float deltaTime)
    {
        foreach (QuantumElementConnection connection in possibleConnections)
        {
            if (Random.value < connection.transitionProbability)
            {
                // If probability condition is met, transition to the connected element's state
                TransitionState(connection.targetElement.probabilityWaveFunction, deltaTime);
            }
        }
    }
}
