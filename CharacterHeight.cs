using System.Collections.Generic;
using UnityEngine;
public class CharacterHeight : MonoBehaviour
{
    public float baseHeight = 1.75f;  // Base height in meters
    private float? manualHeight = null;

    void Start()
    {
        // Ensure that manualHeight is properly initialized
        if (!manualHeight.HasValue)
        {
            manualHeight = baseHeight;
        }
        
        DisplayHeight();  // Display the initial height
    }

    // Method to set the character's height manually
    public void SetManualHeight(float height)
    {
        manualHeight = height;
        Debug.Log($"Manual height set to: {manualHeight} meters");
    }

    // Method to set the character's height using a scaling factor
    public float SetScaledHeight(float scaleFactor)
    {
        float scaledHeight = baseHeight * scaleFactor;
        Debug.Log($"Scaled height based on factor {scaleFactor}: {scaledHeight:F2} meters");
        return scaledHeight;
    }

    // Method to display the current height of the character
    public void DisplayHeight()
    {
        if (manualHeight.HasValue)
        {
            Debug.Log($"Current character height (manual): {manualHeight} meters");
        }
        else
        {
            Debug.Log($"Current character height (base): {baseHeight} meters");
        }
    }
}
