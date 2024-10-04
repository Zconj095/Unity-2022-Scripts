using UnityEngine;

public class ImageProcessor2 : MonoBehaviour
{
    public Texture2D AdjustForMonochrome(Texture2D original)
    {
        Texture2D adjusted = new Texture2D(original.width, original.height);
        for (int x = 0; x < original.width; x++)
        {
            for (int y = 0; y < original.height; y++)
            {
                Color pixel = original.GetPixel(x, y);
                float gray = (pixel.r + pixel.g + pixel.b) / 3;
                adjusted.SetPixel(x, y, new Color(gray, gray, gray));
            }
        }
        adjusted.Apply();
        return adjusted;
    }

    public Texture2D AdjustForNeutralDifficulties(Texture2D original)
    {
        Texture2D adjusted = new Texture2D(original.width, original.height);
        float contrastFactor = 1.2f; // controls the contrast
        float brightnessFactor = -20.0f / 255.0f; // controls the brightness
        for (int x = 0; x < original.width; x++)
        {
            for (int y = 0; y < original.height; y++)
            {
                Color pixel = original.GetPixel(x, y);
                Color adjustedColor = new Color(
                    Mathf.Clamp01(pixel.r * contrastFactor + brightnessFactor),
                    Mathf.Clamp01(pixel.g * contrastFactor + brightnessFactor),
                    Mathf.Clamp01(pixel.b * contrastFactor + brightnessFactor)
                );
                adjusted.SetPixel(x, y, adjustedColor);
            }
        }
        adjusted.Apply();
        return adjusted;
    }

    public Texture2D AdjustForWarmDifficulties(Texture2D original)
    {
        Texture2D adjusted = new Texture2D(original.width, original.height);
        float enhancementFactor = 1.2f; // Increase red component
        float reduceFactor = 0.8f; // Reduce green component slightly if needed
        for (int x = 0; x < original.width; x++)
        {
            for (int y = 0; y < original.height; y++)
            {
                Color pixel = original.GetPixel(x, y);
                Color adjustedColor = new Color(
                    Mathf.Min(pixel.r * enhancementFactor, 1.0f),
                    Mathf.Min(pixel.g * reduceFactor, 1.0f),
                    pixel.b // Unchanged blue component
                );
                adjusted.SetPixel(x, y, adjustedColor);
            }
        }
        adjusted.Apply();
        return adjusted;
    }
}