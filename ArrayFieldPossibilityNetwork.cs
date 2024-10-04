using UnityEngine;

public class ArrayFieldPossibilityNetwork : MonoBehaviour
{
    // Public variables for size, so they can be set in the Unity Inspector
    public int dimensionX = 10;
    public int dimensionY = 10;
    public int dimensionZ = 10;
    public GameObject cubePrefab;  // A prefab for the cubes that will represent the fields

    // Multi-dimensional array representing the field
    private double[,,] possibilityField;

    // Store the cube GameObjects for visualization purposes
    private GameObject[,,] cubeField;

    void Start()
    {
        // Initialize the field with random possibility states
        possibilityField = new double[dimensionX, dimensionY, dimensionZ];
        cubeField = new GameObject[dimensionX, dimensionY, dimensionZ];
        
        InitializeField();
        CreateVisualization();
    }

    // Initialize the field with random values between 0 and 1
    void InitializeField()
    {
        for (int x = 0; x < dimensionX; x++)
        {
            for (int y = 0; y < dimensionY; y++)
            {
                for (int z = 0; z < dimensionZ; z++)
                {
                    possibilityField[x, y, z] = Random.Range(0f, 1f);
                }
            }
        }
    }

    // Create a visual representation of the field using cubes
    void CreateVisualization()
    {
        if (cubePrefab == null)
        {
            Debug.LogError("Cube Prefab is not assigned in the inspector.");
            return;
        }

        for (int x = 0; x < dimensionX; x++)
        {
            for (int y = 0; y < dimensionY; y++)
            {
                for (int z = 0; z < dimensionZ; z++)
                {
                    // Create a cube at the respective field position
                    Vector3 position = new Vector3(x, y, z);
                    cubeField[x, y, z] = Instantiate(cubePrefab, position, Quaternion.identity);
                    
                    // Set cube size based on the possibility value (for visualization)
                    float scaleValue = (float)possibilityField[x, y, z];
                    cubeField[x, y, z].transform.localScale = new Vector3(scaleValue, scaleValue, scaleValue);
                }
            }
        }
    }

    // Function to evolve the field and update the visualization
    void EvolveField()
    {
        double[,,] newField = new double[dimensionX, dimensionY, dimensionZ];

        for (int x = 0; x < dimensionX; x++)
        {
            for (int y = 0; y < dimensionY; y++)
            {
                for (int z = 0; z < dimensionZ; z++)
                {
                    newField[x, y, z] = QuantumEvolutionRule(x, y, z);
                }
            }
        }

        // Update the field and visualization
        possibilityField = newField;
        UpdateVisualization();
    }

    // Update the cubes' sizes to reflect the updated field states
    void UpdateVisualization()
    {
        for (int x = 0; x < dimensionX; x++)
        {
            for (int y = 0; y < dimensionY; y++)
            {
                for (int z = 0; z < dimensionZ; z++)
                {
                    float scaleValue = (float)possibilityField[x, y, z];
                    cubeField[x, y, z].transform.localScale = new Vector3(scaleValue, scaleValue, scaleValue);
                }
            }
        }
    }

    // Simple quantum-inspired evolution rule, averaging neighbor values
    double QuantumEvolutionRule(int x, int y, int z)
    {
        double currentState = possibilityField[x, y, z];
        double sum = currentState;
        int neighbors = 0;

        // Check surrounding neighbors
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int k = -1; k <= 1; k++)
                {
                    if (i == 0 && j == 0 && k == 0) continue;

                    int nx = x + i;
                    int ny = y + j;
                    int nz = z + k;

                    if (nx >= 0 && nx < dimensionX && ny >= 0 && ny < dimensionY && nz >= 0 && nz < dimensionZ)
                    {
                        sum += possibilityField[nx, ny, nz];
                        neighbors++;
                    }
                }
            }
        }

        return sum / (neighbors + 1); // Averaging the neighbors' values
    }

    // Example to call evolve every few seconds (could also use Update for continuous evolution)
    private float evolveTimer = 5f;
    void Update()
    {
        evolveTimer -= Time.deltaTime;
        if (evolveTimer <= 0)
        {
            EvolveField();
            evolveTimer = 5f; // Reset the timer for the next evolution
        }
    }
}
