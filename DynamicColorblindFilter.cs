using UnityEngine;

public class DynamicColorblindFilter : MonoBehaviour
{
    public Texture2D ApplyDynamicColorblindFilter(Texture2D original, string filterType, float intensity = 1.0f)
    {
        Color32[] originalPixels = original.GetPixels32();
        Color32[] adjustedPixels = new Color32[originalPixels.Length];

        switch (filterType)
        {
            case "monochrome":
                adjustedPixels = AdjustForMonochrome(originalPixels, intensity);
                break;
            case "neutral_difficulty":
                adjustedPixels = AdjustForNeutralDifficulty(originalPixels, intensity);
                break;
            case "warm_color_difficulty":
                adjustedPixels = AdjustForWarmColorDifficulty(originalPixels, intensity);
                break;
            case "neutral_greyscale":
                adjustedPixels = AdjustForNeutralGreyScale(originalPixels, intensity);
                break;
            case "warm_greyscale":
                adjustedPixels = AdjustForWarmGreyScale(originalPixels, intensity);
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

    private Color32[] AdjustForMonochrome(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            byte gray = (byte)((pixels[i].r * 0.299 + pixels[i].g * 0.587 + pixels[i].b * 0.114) * intensity);
            adjusted[i] = new Color32(gray, gray, gray, pixels[i].a);
        }
        return adjusted;
    }

    private Color32[] AdjustForNeutralDifficulty(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            byte r = (byte)Mathf.Clamp(pixels[i].r * 1.2f * intensity - 20f * intensity, 0, 255);
            byte g = (byte)Mathf.Clamp(pixels[i].g * 1.2f * intensity - 20f * intensity, 0, 255);
            byte b = (byte)Mathf.Clamp(pixels[i].b * 1.2f * intensity - 20f * intensity, 0, 255);
            adjusted[i] = new Color32(r, g, b, pixels[i].a);
        }
        return adjusted;
    }

    private Color32[] AdjustForWarmColorDifficulty(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            byte r = (byte)Mathf.Clamp((pixels[i].r * 1.2f + pixels[i].g * 0.8f) * intensity, 0, 255);
            adjusted[i] = new Color32(r, pixels[i].g, pixels[i].b, pixels[i].a);
        }
        return adjusted;
    }

    private Color32[] AdjustForNeutralGreyScale(Color32[] pixels, float intensity)
    {
        return AdjustForMonochrome(pixels, intensity); // Placeholder since the action will be similar
    }

    private Color32[] AdjustForWarmGreyScale(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            byte gray = (byte)((pixels[i].r * 0.299 + pixels[i].g * 0.587 + pixels[i].b * 0.114) * 1.1 * intensity);
            adjusted[i] = new Color32(gray, gray, gray, pixels[i].a);
        }
        return adjusted;
    }
}