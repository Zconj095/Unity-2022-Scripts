using UnityEngine;

public class AdvancedColorblindFilter : MonoBehaviour
{
    public Texture2D ApplyAdvancedColorblindFilter(Texture2D original, string filterType, bool contrastMode = false)
    {
        Color32[] originalPixels = original.GetPixels32();
        Color32[] adjustedPixels;

        if (contrastMode)
        {
            adjustedPixels = ApplyHighContrast(originalPixels);
        }
        else
        {
            adjustedPixels = DynamicAdjustmentAlgorithm(originalPixels, filterType, 1.0f); // Default intensity
        }

        Texture2D adjustedImage = new Texture2D(original.width, original.height);
        adjustedImage.SetPixels32(adjustedPixels);
        adjustedImage.Apply();
        return adjustedImage;
    }

    private Color32[] DynamicAdjustmentAlgorithm(Color32[] pixels, string filterType, float intensity)
    {
        Color32[] adjusted = null;
        switch (filterType)
        {
            case "monochrome":
                adjusted = AdjustForMonochrome(pixels, intensity);
                break;
            case "neutral_difficulty":
                adjusted = AdjustForNeutralDifficulty(pixels, intensity);
                break;
            case "warm_color_difficulty":
                adjusted = AdjustForWarmColorDifficulty(pixels, intensity);
                break;
            case "neutral_greyscale":
                adjusted = AdjustForNeutralGreyScale(pixels, intensity);
                break;
            case "warm_greyscale":
                adjusted = AdjustForWarmGreyScale(pixels, intensity);
                break;
            default:
                adjusted = pixels;
                break;
        }

        // Apply intensity scaling to the adjustments
        for (int i = 0; i < pixels.Length; i++)
        {
            adjusted[i] = Color32.Lerp(pixels[i], adjusted[i], intensity);
        }

        return adjusted;
    }

    // Method to apply high contrast adjustments
    private Color32[] ApplyHighContrast(Color32[] pixels)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        float min = 255, max = 0;

        // Compute min and max brightness values
        foreach (var pixel in pixels)
        {
            float gray = pixel.r * 0.299f + pixel.g * 0.587f + pixel.b * 0.114f;
            min = Mathf.Min(gray, min);
            max = Mathf.Max(gray, max);
        }

        float range = max - min;
        // Adjusting the contrast based on computed min and max
        for (int i = 0; i < pixels.Length; i++)
        {
            float gray = pixels[i].r * 0.299f + pixels[i].g * 0.587f + pixels[i].b * 0.114f;
            byte contrast = (byte)(255.0f * ((gray - min) / range));
            adjusted[i] = new Color32(contrast, contrast, contrast, pixels[i].a);
        }

        return adjusted;
    }

    // Adjust pixels for monochrome vision - simulating the absence of color
    private Color32[] AdjustForMonochrome(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            float gray = pixels[i].r * 0.299f + pixels[i].g * 0.587f + pixels[i].b * 0.114f;
            byte grayByte = (byte)gray;
            adjusted[i] = new Color32(
                (byte)Mathf.Lerp(pixels[i].r, grayByte, intensity),
                (byte)Mathf.Lerp(pixels[i].g, grayByte, intensity),
                (byte)Mathf.Lerp(pixels[i].b, grayByte, intensity),
                pixels[i].a
            );
        }

        return adjusted;
    }

    // Adjust pixels for simulations where distinction between colors is reduced (neutral difficulty)
    private Color32[] AdjustForNeutralDifficulty(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            // Slightly reduce the saturation
            float gray = pixels[i].r * 0.299f + pixels[i].g * 0.587f + pixels[i].b * 0.114f;
            adjusted[i] = new Color32(
                (byte)Mathf.Lerp(pixels[i].r, (byte)gray, intensity * 0.5f),
                (byte)Mathf.Lerp(pixels[i].g, (byte)gray, intensity * 0.5f),
                (byte)Mathf.Lerp(pixels[i].b, (byte)gray, intensity * 0.5f),
                pixels[i].a
            );
        }

        return adjusted;
    }

    // Simulating warmth in colors (for warm color difficulty)
    private Color32[] AdjustForWarmColorDifficulty(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            // Enhance red, slightly reduce blue
            adjusted[i] = new Color32(
                (byte)Mathf.Clamp(pixels[i].r + 30 * intensity, 0, 255),
                pixels[i].g,
                (byte)Mathf.Clamp(pixels[i].b - 30 * intensity, 0, 255),
                pixels[i].a
            );
        }

        return adjusted;
    }

    // Adjust pixels to simulate neutral greyscale tones
    private Color32[] AdjustForNeutralGreyScale(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            float gray = (pixels[i].r + pixels[i].g + pixels[i].b) / 3;
            byte grayByte = (byte)gray;
            adjusted[i] = new Color32(grayByte, grayByte, grayByte, pixels[i].a);
        }

        return adjusted;
    }

    // Adjust pixels to simulate warm greyscale tones (slightly yellowish tint)
    private Color32[] AdjustForWarmGreyScale(Color32[] pixels, float intensity)
    {
        Color32[] adjusted = new Color32[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            float gray = (pixels[i].r + pixels[i].g + pixels[i].b) / 3;
            byte grayByte = (byte)gray;
            // Adding a slight yellow tint to the greyscale
            adjusted[i] = new Color32(
                (byte)Mathf.Lerp(grayByte, pixels[i].r, intensity * 0.2f),
                (byte)Mathf.Lerp(grayByte, pixels[i].g, intensity * 0.2f),
                grayByte,
                pixels[i].a
            );
        }

        return adjusted;
    }
}