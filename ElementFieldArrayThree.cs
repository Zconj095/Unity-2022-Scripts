using UnityEngine;

public class ElementFieldArrayThree: MonoBehaviour
{
    public QuantumElementThree[,] fieldArray; // 2D array of elements
    public int sizeX, sizeY; // Dimensions of the element field

    public ElementFieldArrayThree(int x, int y)
    {
        sizeX = x;
        sizeY = y;
        fieldArray = new QuantumElementThree[sizeX, sizeY];
        InitializeField();
        BuildPossibilityNetwork();
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
                
                // Randomly assign an elemental attribute
                ElementalTypeTwo randomElement = (ElementalTypeTwo)UnityEngine.Random.Range(0, System.Enum.GetValues(typeof(ElementalTypeTwo)).Length);
                float elementalIntensity = UnityEngine.Random.Range(0.1f, 1f);

                fieldArray[i, j] = new QuantumElementThree(randomPosition, randomEnergy, initialWaveFunction, hyperCoord, randomElement, elementalIntensity);
            }
        }
    }

    // Build the possibility network by connecting nearby elements with transition probabilities
    private void BuildPossibilityNetwork()
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElementThree element = fieldArray[i, j];

                // Connect to neighboring elements (left, right, up, down)
                if (i > 0) AddConnectionBetweenElements(element, fieldArray[i - 1, j]);
                if (i < sizeX - 1) AddConnectionBetweenElements(element, fieldArray[i + 1, j]);
                if (j > 0) AddConnectionBetweenElements(element, fieldArray[i, j - 1]);
                if (j < sizeY - 1) AddConnectionBetweenElements(element, fieldArray[i, j + 1]);
            }
        }
    }

    // Add a connection between two elements with a random probability
    private void AddConnectionBetweenElements(QuantumElementThree elementA, QuantumElementThree elementB)
    {
        float probability = UnityEngine.Random.Range(0.0f, 1.0f); // Random probability between 0 and 1
        elementA.AddConnection(elementB, probability);
        elementB.AddConnection(elementA, probability); // Bidirectional connection
    }

    // Simulate the field evolution based on the possibility network
    public void SimulateFieldEvolution(float deltaTime)
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElementThree element = fieldArray[i, j];
                element.TransitionBasedOnNetwork(deltaTime);
            }
        }
    }

    // Visualize the field (showing possible connections with lines, optionally)
    public void VisualizeField()
    {
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                QuantumElementThree element = fieldArray[i, j];
                Color color = ElementalTypeToColor(element.elementalAttribute);
                Debug.DrawLine(element.position, element.position + Vector3.up * 0.1f, color);

                // Optionally visualize connections (Debug lines between connected elements)
                foreach (var connection in element.possibleConnections)
                {
                    Debug.DrawLine(element.position, connection.targetElement.position, Color.yellow);
                }
            }
        }
    }

    // Helper function to convert elemental type to a color (same as before)
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
