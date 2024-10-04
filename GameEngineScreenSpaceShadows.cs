using System.Collections.Generic;
using UnityEngine;

public class GameEngineScreenSpaceShadows : MonoBehaviour
{
    private bool enabled = true;  // Default is enabled
    private string quality = "High";  // Default quality

    public (bool enabled, string quality) ScreenSpaceShadows
    {
        get { return (enabled, quality); }
        set
        {
            enabled = value.enabled;
            if (value.quality == "Low" || value.quality == "Medium" || value.quality == "High")
            {
                quality = value.quality;
                Debug.Log($"Screen Space Shadows set to: {(enabled ? "enabled" : "disabled")} ({quality} quality)");
            }
            else
            {
                Debug.LogError("Invalid quality setting. Choose 'Low', 'Medium', or 'High'.");
            }
        }
    }

    public void DisplayInfo()
    {
        Debug.Log($"Screen Space Shadows: {(enabled ? "enabled" : "disabled")} ({quality} quality)");
    }

    void Start()
    {
        DisplayInfo();
    }
}
