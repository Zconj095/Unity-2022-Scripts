using UnityEngine;

public class ColorblindFilter : MonoBehaviour
{
    public Texture2D ApplyColorblindFilter(Texture2D original, string filterType)
    {
        Color32[] originalPixels = original.GetPixels32();
        Color32[] adjustedPixels = new Color32[originalPixels.Length];

        switch(filterType)
        {
            case "protanopia":
                adjustedPixels = AdjustForProtanopia(originalPixels);
                break;
            case "deuteranopia":
                adjustedPixels = AdjustForDeuteranopia(originalPixels);
                break;
            case "tritanopia":
                adjustedPixels = AdjustForTritanopia(originalPixels);
                break;
            default:
                adjustedPixels = originalPixels;
                break;
        }

        Texture2D adjustedImage = new Texture2D(original.width, original.height);
        adjustedImage.SetPixels32(adjustedPixels);
        adjustedImage.Apply();
        return adjustedImage;
    }

    private Color32[] AdjustForDeuteranopia(Color32[] pixels)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for(int i = 0; i < pixels.Length; i++)
        {
            float r = pixels[i].r * 0.625f + pixels[i].g * 0.375f;
            float g = pixels[i].r * 0.70f + pixels[i].g * 0.30f;
            float b = pixels[i].g * 0.30f + pixels[i].b * 0.70f;
            adjusted[i] = new Color32((byte)r, (byte)g, (byte)b, pixels[i].a);
        }
        return adjusted;
    }

    private Color32[] AdjustForTritanopia(Color32[] pixels)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for(int i = 0; i < pixels.Length; i++)
        {
            float r = pixels[i].r * 0.95f + pixels[i].g * 0.05f;
            float g = pixels[i].r * 0.433f + pixels[i].b * 0.567f;
            float b = pixels[i].g * 0.475f + pixels[i].b * 0.525f;
            adjusted[i] = new Color32((byte)r, (byte)g, (byte)b, pixels[i].a);
        }
        return adjusted;
    }

    private Color32[] AdjustForProtanopia(Color32[] pixels)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for(int i = 0; i < pixels.Length; i++)
        {
            float r = pixels[i].g * 0.567f + pixels[i].b * 0.433f;
            float g = pixels[i].g * 0.558f + pixels[i].b * 0.442f;
            float b = pixels[i].r * 0.242f + pixels[i].b * 0.758f;
            adjusted[i] = new Color32((byte)r, (byte)g, (byte)b, pixels[i].a);
        }
        return adjusted;
    }
}