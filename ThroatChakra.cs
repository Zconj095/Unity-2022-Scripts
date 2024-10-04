using UnityEngine;

public class ThroatChakra : MonoBehaviour
{
    public Color chakraColor = Color.cyan; // Cyan for Throat Chakra
    public float chakraTransparency = 1.0f;

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
