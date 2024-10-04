using UnityEngine;

public class CrownChakra : MonoBehaviour
{
    public Color chakraColor = new Color(0.8f, 0.4f, 1f); // Light violet for Crown Chakra
    public float chakraTransparency = 1.0f; // Fully opaque

    private Material chakraMaterial;

    void Start()
    {
        // Ensure the chakra has a renderer and assign a new material
        Renderer renderer = GetComponent<Renderer>();
        if (renderer == null)
        {
            Debug.LogError("Renderer not found on " + gameObject.name);
            return;
        }

        // Create a new material and assign it to the renderer
        chakraMaterial = new Material(Shader.Find("Standard"));
        renderer.material = chakraMaterial;

        // Set the color and transparency
        SetChakraMaterial();
    }

    void SetChakraMaterial()
    {
        if (chakraMaterial != null)
        {
            chakraMaterial.color = chakraColor;

            // Adjust transparency
            Color materialColor = chakraMaterial.color;
            materialColor.a = chakraTransparency;
            chakraMaterial.color = materialColor;

            // Ensure it's set to transparent rendering mode if needed
            if (chakraTransparency < 1.0f)
            {
                SetMaterialTransparent(chakraMaterial);
            }
        }
    }

    void SetMaterialTransparent(Material material)
    {
        // Make the material render as transparent
        material.SetFloat("_Mode", 3);  // 3 is the Transparent mode in Unity's Standard Shader
        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        material.SetInt("_ZWrite", 0);
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }
}
