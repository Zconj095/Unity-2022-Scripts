using UnityEngine;

public class ImageAdjustment : MonoBehaviour
{
    // Adjust a Texture2D for simple color difficulties
    public Texture2D AdjustForSimpleColorDifficulty(Texture2D original)
    {
        Texture2D adjusted = new Texture2D(original.width, original.height);
        for (int x = 0; x < original.width; x++)
        {
            for (int y = 0; y < original.height; y++)
            {
                Color pixel = original.GetPixel(x, y);
                // Convert RGB to HSV
                Color.RGBToHSV(pixel, out float H, out float S, out float V);
                // Enhancing saturation and value
                S *= 1.2f;
                V *= 1.1f;
                // Clamping S and V to avoid exceeding bounds
                S = Mathf.Clamp01(S);
                V = Mathf.Clamp01(V);
                // Convert back to RGB
                Color adjustedColor = ColorUtils.HSVToRGB(H, S, V);
                adjusted.SetPixel(x, y, adjustedColor);
            }
        }
        adjusted.Apply();
        return adjusted;
    }
}