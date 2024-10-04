using UnityEngine;
using System.Collections.Generic; // Added to use Dictionary

public class ColorAdjuster : MonoBehaviour
{
    public Texture2D ApplyColorAdjustments(Texture2D original, Dictionary<string, float> adjustments)
    {
        Color32[] pixels = original.GetPixels32();
        
        for (int i = 0; i < pixels.Length; i++)
        {
            if (adjustments.ContainsKey("red_shift"))
                pixels[i].r = (byte)Mathf.Clamp(pixels[i].r * adjustments["red_shift"], 0, 255);
            if (adjustments.ContainsKey("green_shift"))
                pixels[i].g = (byte)Mathf.Clamp(pixels[i].g * adjustments["green_shift"], 0, 255);
            if (adjustments.ContainsKey("blue_shift"))
                pixels[i].b = (byte)Mathf.Clamp(pixels[i].b * adjustments["blue_shift"], 0, 255);
        }

        Texture2D adjustedImage = new Texture2D(original.width, original.height);
        adjustedImage.SetPixels32(pixels);
        adjustedImage.Apply();
        return adjustedImage;
    }
}