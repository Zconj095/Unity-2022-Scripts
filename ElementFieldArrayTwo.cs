using UnityEngine;

public class ElementFieldArrayTwo: MonoBehaviour
{
    public QuantumElementTwo[,] fieldArray; // 2D array of elements
    public int sizeX, sizeY; // Dimensions of the element field

    // Constructor - initializes the field array (no return type needed for constructors)
    public ElementFieldArrayTwo(int x, int y)
    {
        sizeX = x;
        sizeY = y;
        fieldArray = new QuantumElementTwo[sizeX, sizeY];
        InitializeField();
    }

    // Method to initialize the field - doesn't return anything, so it should be 'void'
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
                
                // Randomly assign an elemental attribute
                ElementalTypeTwo randomElement = (ElementalTypeTwo)UnityEngine.Random.Range(0, System.Enum.GetValues(typeof(ElementalTypeTwo)).Length);
                float elementalIntensity = UnityEngine.Random.Range(0.1f, 1f);

                fieldArray[i, j] = new QuantumElementTwo(randomPosition, randomEnergy, initialWaveFunction, hyperCoord, randomElement, elementalIntensity);
            }
        }
    }

    // Method to simulate field evolution over time (void, as it modifies the state)
    public void SimulateFieldEvolution(float deltaTime)
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElementTwo element = fieldArray[i, j];
                Complex newWaveFunction = new Complex(UnityEngine.Random.value, UnityEngine.Random.value);
                element.TransitionState(newWaveFunction, deltaTime);
            }
        }
    }

    // Method to simulate interactions between neighboring elements (void, as it modifies the state)
    public void SimulateElementalInteractions()
    {
        for (int i = 0; i < sizeX - 1; i++)
        {
            for (int j = 0; j < sizeY - 1; j++)
            {
                QuantumElementTwo currentElement = fieldArray[i, j];
                QuantumElementTwo rightNeighbor = fieldArray[i + 1, j];
                QuantumElementTwo downNeighbor = fieldArray[i, j + 1];
                
                // Simulate interaction with neighboring elements
                currentElement.InteractWith(rightNeighbor);
                currentElement.InteractWith(downNeighbor);
            }
        }
    }

    // Method to visualize the field (void, as it does not return anything, just draws debug lines)
    public void VisualizeField()
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElementTwo element = fieldArray[i, j];
                Color color = ElementalTypeToColor(element.elementalAttribute);
                Debug.DrawLine(element.position, element.position + Vector3.up * 0.1f, color);
            }
        }
    }

    // Helper function to convert elemental type to a color (returns Color)
    private Color ElementalTypeToColor(ElementalTypeTwo type)
    {
        switch (type)
        {
            case ElementalTypeTwo.Fire: return Color.red;
            case ElementalTypeTwo.Water: return Color.blue;
            case ElementalTypeTwo.Earth: return Color.green;
            case ElementalTypeTwo.Air: return Color.white;
            case ElementalTypeTwo.Lightning: return Color.yellow;
            case ElementalTypeTwo.Ice: return Color.cyan;
            default: return Color.gray;
        }
    }
}
