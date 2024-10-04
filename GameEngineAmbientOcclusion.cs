using System.Collections.Generic;
using UnityEngine;
public class GameEngineAmbientOcclusion : MonoBehaviour
{
    public string level = "Medium"; // AO quality level ('Low', 'Medium', 'High', 'Ultra')
    private AOSettings aoSettings;

    [System.Serializable]
    public class AOSettings
    {
        public string intensity;
        public string quality;
        public string description;

        public AOSettings(string intensity, string quality, string description)
        {
            this.intensity = intensity;
            this.quality = quality;
            this.description = description;
        }
    }

    void Start()
    {
        // Ensure that the AO settings are initialized based on the current level
        aoSettings = DetermineAOSettings(level);
        DisplayInfo();
    }

    private AOSettings DetermineAOSettings(string level)
    {
        // Define AO settings based on level
        switch (level)
        {
            case "Low":
                return new AOSettings("Subtle", "Basic", "Minimal AO effect, optimized for performance.");
            case "High":
                return new AOSettings("Pronounced", "High", "Strong AO effect with enhanced realism, higher resource usage.");
            case "Ultra":
                return new AOSettings("Very Pronounced", "Very High", "Maximum AO effect with the highest realism, significant performance cost.");
            case "Medium":
            default:
                return new AOSettings("Balanced", "Standard", "Balanced AO effect with moderate realism and performance.");
        }
    }

    public AOSettings AmbientOcclusionSettings
    {
        get { return aoSettings; }
        set
        {
            if (value != null)
            {
                level = value.quality; // Assuming quality is used as level in this case
                aoSettings = value;
            }
            else
            {
                Debug.LogWarning("Attempted to set AmbientOcclusionSettings to null.");
            }
        }
    }

    public void AdjustAO(string performancePreference)
    {
        switch (performancePreference.ToLower())
        {
            case "low":
                AmbientOcclusionSettings = DetermineAOSettings("Low");
                break;
            case "high":
                AmbientOcclusionSettings = DetermineAOSettings("High");
                break;
            case "medium":
            default:
                AmbientOcclusionSettings = DetermineAOSettings("Medium");
                break;
        }
    }

    public void DisplayInfo()
    {
        if (aoSettings != null)
        {
            Debug.Log($"Ambient Occlusion Level: {level}");
            Debug.Log($"Intensity: {aoSettings.intensity}");
            Debug.Log($"Quality: {aoSettings.quality}");
            Debug.Log($"Description: {aoSettings.description}");
        }
        else
        {
            Debug.LogError("AO Settings are missing or not initialized.");
        }
    }
}
