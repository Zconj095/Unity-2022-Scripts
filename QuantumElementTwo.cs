using UnityEngine;

public class QuantumElementTwo: MonoBehaviour
{
    public Vector3 position;  // Spatial coordinate in 3D space
    public float energyLevel; // Energy associated with this quantum element
    public Complex probabilityWaveFunction; // Quantum state of the element
    public Vector4 hyperDimensionalCoordinate; // Represents extra-dimensional influence
    
    public ElementalTypeTwo elementalAttribute;  // Elemental type (e.g., Fire, Water)
    public float elementalIntensity;          // Intensity of the elemental power

    // Constructor to initialize the quantum element
    public QuantumElementTwo(Vector3 pos, float energy, Complex waveFunc, Vector4 hyperDimCoord, ElementalTypeTwo element, float intensity)
    {
        position = pos;
        energyLevel = energy;
        probabilityWaveFunction = waveFunc;
        hyperDimensionalCoordinate = hyperDimCoord;
        elementalAttribute = element;
        elementalIntensity = intensity;
    }

    // Method to simulate quantum state transitions
    public void TransitionState(Complex newWaveFunc, float timeDelta)
    {
        probabilityWaveFunction = Complex.Lerp(probabilityWaveFunction, newWaveFunc, timeDelta);
    }

    // Function to calculate probability distribution
    public float CalculateProbabilityDensity()
    {
        return probabilityWaveFunction.MagnitudeSquared(); // |ψ|^2 gives us probability density
    }

    // Simulate elemental interaction with another element
    public void InteractWith(QuantumElementTwo other)
    {
        // Example rule: Fire loses to Water, Water loses to Earth, etc.
        if (elementalAttribute == ElementalTypeTwo.Fire && other.elementalAttribute == ElementalTypeTwo.Water)
        {
            energyLevel -= other.elementalIntensity;  // Fire loses energy to Water
        }
        else if (elementalAttribute == ElementalTypeTwo.Water && other.elementalAttribute == ElementalTypeTwo.Fire)
        {
            energyLevel += other.elementalIntensity;  // Water gains energy from Fire
        }
        // Add more interaction rules here...
    }
}
