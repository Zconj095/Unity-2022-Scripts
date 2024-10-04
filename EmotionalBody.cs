using UnityEngine;

public class EmotionalBody : MonoBehaviour
{
    public Color auraColor = new Color(1f, 0.7f, 0.7f); // Light red for Emotional Body
    public float auraTransparency = 0.3f;

    private Material auraMaterial;

    void Start()
    {
        Renderer renderer = GetComponent<Renderer>();
        auraMaterial = new Material(Shader.Find("Standard"));
        renderer.material = auraMaterial;
        SetAuraMaterial();
    }

    void SetAuraMaterial()
    {
        auraMaterial.color = auraColor;
        auraMaterial.color = new Color(auraColor.r, auraColor.g, auraColor.b, auraTransparency);
        if (auraTransparency < 1.0f) SetMaterialTransparent(auraMaterial);
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
