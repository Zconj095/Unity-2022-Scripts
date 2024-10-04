using UnityEngine;

public class AuricLayerGenerator : MonoBehaviour
{
    public GameObject emptyObject; // Assign a parent object for the body (for hierarchy purposes)
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
        if (emptyObject == null)
        {
            Debug.LogError("Empty object not assigned!");
            return;
        }

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
        // Create a new empty object for this specific auric layer
        GameObject auricLayerContainer = new GameObject(auricLayerNames[index] + " Container");
        auricLayerContainer.transform.parent = emptyObject.transform; // Parent to the main emptyObject
        auricLayerContainer.transform.localPosition = Vector3.zero; // Position at the center of the body

        // Create a new GameObject for the auric layer as a sphere
        GameObject auricLayer = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        auricLayer.transform.parent = auricLayerContainer.transform; // Parent the layer to its own container
        auricLayer.transform.localPosition = Vector3.zero; // Center it within its own container

        // Scale each auric layer progressively larger
        float scaleMultiplier = baseScale + (index * scaleIncrement);
        auricLayer.transform.localScale = new Vector3(scaleMultiplier, scaleMultiplier, scaleMultiplier);

        // Set the material color and transparency
        Renderer renderer = auricLayer.GetComponent<Renderer>();
        renderer.material = new Material(Shader.Find("Standard"));
        renderer.material.color = auricLayerColors[index];

        // Set transparency of the auric layer
        SetMaterialTransparent(renderer.material);
        Color layerColor = renderer.material.color;
        layerColor.a = auraTransparency; // Set alpha for transparency
        renderer.material.color = layerColor;
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
