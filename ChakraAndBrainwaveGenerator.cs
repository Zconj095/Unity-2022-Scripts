using UnityEngine;

public class ChakraAndBrainwaveGenerator : MonoBehaviour
{
    // Variables for chakras (assuming they already exist)
    // public GameObject emptyObject; 
    // ... (existing code for chakras)

    public GameObject brainwaveEmptyObject; // Assign an empty object for brainwaves
    private string[] brainwaveClasses = { "Delta", "Theta", "Alpha", "Beta", "Gamma" };
    private (float, float)[] brainwaveRanges = { (0.5f, 4f), (4f, 8f), (8f, 12f), (12f, 30f), (30f, 100f) };
    private const int numBrainwaves = 1000;

    // Variables for visual representation
    public float minBrainwaveSize = 0.2f; // Minimum size for the brainwave sphere
    public float maxBrainwaveSize = 1.0f; // Maximum size for the brainwave sphere

    void Start()
    {
        // Existing code to generate chakras (if any)

        if (brainwaveEmptyObject == null)
        {
            Debug.LogError("Brainwave empty object not assigned!");
            return;
        }

        GenerateBrainwaves();
    }

    void GenerateBrainwaves()
    {
        for (int i = 0; i < numBrainwaves; i++)
        {
            // Select random brainwave class
            int classIndex = Random.Range(0, brainwaveClasses.Length);
            float frequency = Random.Range(brainwaveRanges[classIndex].Item1, brainwaveRanges[classIndex].Item2);
            frequency = Mathf.Round(frequency * 10000f) / 10000f; // 4 decimal places

            // Create the brainwave object
            CreateBrainwave(brainwaveClasses[classIndex], frequency);
        }
    }

    void CreateBrainwave(string className, float frequency)
    {
        // Create a new GameObject for the brainwave
        GameObject brainwave = GameObject.CreatePrimitive(PrimitiveType.Sphere); // Visual representation as a sphere
        brainwave.transform.parent = brainwaveEmptyObject.transform;

        // Customize the brainwave object
        brainwave.name = className + ": " + frequency.ToString("F4") + " Hz";
        
        // Set size of brainwave based on frequency range
        float normalizedFrequency = Mathf.InverseLerp(brainwaveRanges[0].Item1, brainwaveRanges[4].Item2, frequency);
        float scale = Mathf.Lerp(minBrainwaveSize, maxBrainwaveSize, normalizedFrequency);
        brainwave.transform.localScale = new Vector3(scale, scale, scale);

        // Assign random position for visual variety
        brainwave.transform.localPosition = new Vector3(
            Random.Range(-5f, 5f), 
            Random.Range(-5f, 5f), 
            Random.Range(-5f, 5f)
        );

        // Set brainwave color based on class
        Renderer renderer = brainwave.GetComponent<Renderer>();
        renderer.material = new Material(Shader.Find("Standard"));

        switch (className)
        {
            case "Delta":
                renderer.material.color = Color.blue; // Delta brainwaves are usually associated with deeper blue tones
                break;
            case "Theta":
                renderer.material.color = Color.magenta; // Theta brainwaves could be represented with magenta
                break;
            case "Alpha":
                renderer.material.color = Color.green; // Alpha brainwaves can have green tones
                break;
            case "Beta":
                renderer.material.color = Color.yellow; // Beta waves are often represented in yellow
                break;
            case "Gamma":
                renderer.material.color = Color.red; // Gamma brainwaves can be represented in red
                break;
        }

        // Add other components to brainwave if needed (e.g., Rigidbody, scripts)
    }
}
