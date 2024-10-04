using UnityEngine;
using System;

public class ElementFieldArray: MonoBehaviour
{
    public QuantumElement[,] fieldArray; // 2D array of elements for simplicity
    public int sizeX, sizeY; // Dimensions of the element field

    public ElementFieldArray(int x, int y)
    {
        sizeX = x;
        sizeY = y;
        fieldArray = new QuantumElement[sizeX, sizeY];
        InitializeField();
    }

    // Initialize the field with random quantum elements
    private void InitializeField()
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                Vector3 randomPosition = new Vector3(i, j, 0);
                float randomEnergy = UnityEngine.Random.Range(0f, 100f);
                Complex initialWaveFunction = new Complex(UnityEngine.Random.value, UnityEngine.Random.value);
                Vector4 hyperCoord = new Vector4(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);

                fieldArray[i, j] = new QuantumElement(randomPosition, randomEnergy, initialWaveFunction, hyperCoord);
            }
        }
    }

    // Simulate field evolution over time
    public void SimulateFieldEvolution(float deltaTime)
    {
        // Loop through each element and evolve its state
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElement element = fieldArray[i, j];
                // Randomly change the wave function (simplified quantum state transition)
                Complex newWaveFunction = new Complex(UnityEngine.Random.value, UnityEngine.Random.value);
                element.TransitionState(newWaveFunction, deltaTime);
            }
        }
    }

    // Calculate the total field energy by summing the energy levels of all elements
    public float CalculateTotalFieldEnergy()
    {
        float totalEnergy = 0;
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                totalEnergy += fieldArray[i, j].energyLevel;
            }
        }
        return totalEnergy;
    }

    // Visualize the field (this could use Unity's particle system or some visual representation)
    public void VisualizeField()
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElement element = fieldArray[i, j];
                Debug.DrawLine(element.position, element.position + Vector3.up * 0.1f, Color.Lerp(Color.blue, Color.red, element.energyLevel / 100f));
            }
        }
    }
}
