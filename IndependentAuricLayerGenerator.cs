using UnityEngine;

public class IndependentAuricLayerGenerator : MonoBehaviour
{
    // Define separate empty objects for each auric layer
    public GameObject[] auraObjects = new GameObject[7]; // Empty game objects for each auric layer (assigned in the Inspector)

    public float baseScale = 1.5f; // Base scale for the Etheric layer
    public float scaleIncrement = 1.0f; // How much larger each subsequent layer should be
    public float auraTransparency = 0.3f; // Transparency of the auric layers

    private string[] auricLayerNames = {
        "Etheric Body",
        "Emotional Body",
        "Mental Body",
        "Astral Body",
        "Etheric Template",
        "Celestial Body",
        "Causal Body"
    };

    private Color[] auricLayerColors = {
        new Color(0.9f, 0.9f, 1f), // Etheric Body (light blue)
        new Color(1f, 0.7f, 0.7f), // Emotional Body (light red/pink)
        new Color(1f, 1f, 0.6f),   // Mental Body (light yellow)
        new Color(0.6f, 1f, 0.6f), // Astral Body (light green)
        new Color(0.7f, 0.7f, 1f), // Etheric Template (light purple/blue)
        new Color(0.9f, 0.6f, 1f), // Celestial Body (light purple)
        new Color(1f, 1f, 1f)      // Causal Body (white)
    };

    void Start()
    {
        // Ensure each aura object is assigned
        for (int i = 0; i < auraObjects.Length; i++)
        {
            if (auraObjects[i] == null)
            {
                Debug.LogError(auricLayerNames[i] + " empty object not assigned!");
                return;
            }
        }

        // Generate the auric layers
        GenerateAuricLayers();
    }

    void GenerateAuricLayers()
    {
        for (int i = 0; i < auricLayerNames.Length; i++)
        {
            CreateAuricLayer(i);
        }
    }

    void CreateAuricLayer(int index)
    {
        // Create a new GameObject for the auric layer as a sphere
        GameObject auricLayer = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        auricLayer.transform.parent = auraObjects[index].transform;

        // Set the auric layer's position relative to its parent aura object
        auricLayer.transform.localPosition = Vector3.zero;

        // Scale each auric layer progressively larger
        float scaleMultiplier = baseScale + (index * scaleIncrement);
        auricLayer.transform.localScale = new Vector3(scaleMultiplier, scaleMultiplier, scaleMultiplier);

        // Set the name of the auric layer
        auricLayer.name = auricLayerNames[index];

        // Create and assign a unique material to each auric layer
        Renderer renderer = auricLayer.GetComponent<Renderer>();
        Material newMaterial = CreateMaterialWithShader(auricLayerColors[index]);

        // Assign the new material to the auric layer's renderer
        renderer.material = newMaterial;
    }

    Material CreateMaterialWithShader(Color color)
    {
        // Create a new material using Unity's Standard Shader
        Shader standardShader = Shader.Find("Standard");
        if (standardShader == null)
        {
            Debug.LogError("Standard shader not found!");
            return null;
        }

        Material material = new Material(standardShader);
        material.color = color;

        // Make the material transparent
        SetMaterialTransparent(material);
        Color materialColor = material.color;
        materialColor.a = auraTransparency; // Set the transparency
        material.color = materialColor;

        return material;
    }

    void SetMaterialTransparent(Material material)
    {
        // Set the rendering mode of the material to Transparent for proper transparency rendering
        material.SetFloat("_Mode", 3);  // 3 is the mode for Transparent
        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        material.SetInt("_ZWrite", 0);
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }
}
