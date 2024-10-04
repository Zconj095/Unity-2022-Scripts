using UnityEngine;

public class ChakraGenerator : MonoBehaviour
{
    public GameObject emptyObject; // Assign your empty object in the inspector
    public float chakraSpacing = 1.0f; // Spacing between each chakra along the Y-axis
    public float auraScaleFactor = 2.0f; // Scale factor for the size of the aura surrounding the empty object
    public float auraTransparency = 0.3f; // Transparency of the aura

    private Color[] chakraColors = {
        Color.red,                    // Root
        new Color(1f, 0.6f, 0f),      // Sacral
        Color.yellow,                 // Solar Plexus
        Color.green,                  // Heart
        Color.cyan,                   // Throat
        new Color(0.5f, 0f, 1f),      // Third Eye
        new Color(0.8f, 0.4f, 1f)     // Crown
    };

    private string[] chakraNames = {
        "Root Chakra",
        "Sacral Chakra",
        "Solar Plexus Chakra",
        "Heart Chakra",
        "Throat Chakra",
        "Third Eye Chakra",
        "Crown Chakra"
    };

    void Start()
    {
        if (emptyObject == null)
        {
            Debug.LogError("Empty object not assigned!");
            return;
        }

        // Create the chakras aligned along the Y-axis inside the body
        for (int i = 0; i < chakraColors.Length; i++)
        {
            CreateChakra(i);
        }

        // Create the aura around the body, connected to the empty object
        CreateAura();
    }

    void CreateChakra(int index)
    {
        // Create the chakra sphere, aligned with the inner body
        GameObject chakra = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        chakra.transform.parent = emptyObject.transform;
        chakra.transform.localPosition = new Vector3(0f, index * chakraSpacing, 0f); // Position along the Y-axis
        Renderer chakraRenderer = chakra.GetComponent<Renderer>();
        chakraRenderer.material = new Material(Shader.Find("Standard"));
        chakraRenderer.material.color = chakraColors[index];
        chakra.name = chakraNames[index];
    }

    void CreateAura()
    {
        // Create a single aura around the empty object, representing the outer energy field
        GameObject aura = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        aura.transform.parent = emptyObject.transform;
        aura.transform.localPosition = Vector3.zero; // Aura is centered on the empty object
        aura.transform.localScale = new Vector3(auraScaleFactor, auraScaleFactor, auraScaleFactor); // Aura surrounds the object

        // Set aura color to a neutral energy color (could be customizable)
        Renderer auraRenderer = aura.GetComponent<Renderer>();
        auraRenderer.material = new Material(Shader.Find("Standard"));
        auraRenderer.material.color = Color.white; // Neutral color

        // Make the aura semi-transparent
        SetMaterialTransparent(auraRenderer.material);
        Color auraColor = auraRenderer.material.color;
        auraColor.a = auraTransparency; // Set transparency
        auraRenderer.material.color = auraColor;

        aura.name = "Energy Aura";
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
