using UnityEngine;

public class ImageProcessor : MonoBehaviour
{
    public Texture2D AdjustForProtanopia(Texture2D original)
    {
        Texture2D adjusted = new Texture2D(original.width, original.height);
        for (int x = 0; x < original.width; x++)
        {
            for (int y = 0; y < original.height; y++)
            {
                Color pixel = original.GetPixel(x, y);
                // Example transformation for protanopia (values are illustrative)
                float newRed = pixel.r * 0.567f + pixel.g * 0.433f;
                float newGreen = pixel.g * 0.558f + pixel.r * 0.442f;
                float newBlue = pixel.b * 0.758f + pixel.g * 0.242f;
                adjusted.SetPixel(x, y, new Color(newRed, newGreen, newBlue));
            }
        }
        adjusted.Apply();
        return adjusted;
    }
}