using UnityEngine;

public class SacralChakra : MonoBehaviour
{
    public Color chakraColor = new Color(1f, 0.6f, 0f); // Orange for Sacral Chakra
    public float chakraTransparency = 1.0f; // Fully opaque

    private Material chakraMaterial;

    void Start()
    {
        Renderer renderer = GetComponent<Renderer>();
        chakraMaterial = new Material(Shader.Find("Standard"));
        renderer.material = chakraMaterial;
        SetChakraMaterial();
    }

    void SetChakraMaterial()
    {
        chakraMaterial.color = chakraColor;
        chakraMaterial.color = new Color(chakraColor.r, chakraColor.g, chakraColor.b, chakraTransparency);
        if (chakraTransparency < 1.0f) SetMaterialTransparent(chakraMaterial);
    }

    void SetMaterialTransparent(Material material)
    {
        material.SetFloat("_Mode", 3);
        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        material.SetInt("_ZWrite", 0);
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }
}
