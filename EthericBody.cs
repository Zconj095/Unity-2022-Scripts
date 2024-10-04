using UnityEngine;

public class EthericBody : MonoBehaviour
{
    public Color auraColor = new Color(0.9f, 0.9f, 1f); // Light blue for Etheric Body
    public float auraTransparency = 0.3f; // Semi-transparent by default

    private Material auraMaterial;

    void Start()
    {
        // Ensure the aura has a renderer and assign a new material
        Renderer renderer = GetComponent<Renderer>();
        if (renderer == null)
        {
            Debug.LogError("Renderer not found on " + gameObject.name);
            return;
        }

        // Create a new material and assign it to the renderer
        auraMaterial = new Material(Shader.Find("Standard"));
        renderer.material = auraMaterial;

        // Set the color and transparency
        SetAuraMaterial();
    }

    void SetAuraMaterial()
    {
        if (auraMaterial != null)
        {
            auraMaterial.color = auraColor;

            // Adjust transparency
            Color materialColor = auraMaterial.color;
            materialColor.a = auraTransparency;
            auraMaterial.color = materialColor;

            // Ensure it's set to transparent rendering mode
            if (auraTransparency < 1.0f)
            {
                SetMaterialTransparent(auraMaterial);
            }
        }
    }

    void SetMaterialTransparent(Material material)
    {
        // Similar transparent material logic as the chakras
        material.SetFloat("_Mode", 3);
        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        material.SetInt("_ZWrite", 0);
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }
}
