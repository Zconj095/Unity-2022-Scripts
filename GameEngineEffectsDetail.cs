using UnityEngine;

public class GameEngineEffectsDetail : MonoBehaviour
{
    [SerializeField]
    private string level = "Medium";  // Default effects detail level

    private EffectsDetailSettings effectsSettings;

    // Class to represent effects detail settings
    private class EffectsDetailSettings
    {
        public string complexity;
        public string resourceUsage;
        public string description;

        public EffectsDetailSettings(string complexity, string resourceUsage, string description)
        {
            this.complexity = complexity;
            this.resourceUsage = resourceUsage;
            this.description = description;
        }
    }

    void Start()
    {
        effectsSettings = DetermineEffectsDetail(level);
        DisplayInfo();  // Display initial settings
    }

    // Method to determine effects detail settings based on quality level
    private EffectsDetailSettings DetermineEffectsDetail(string level)
    {
        switch (level)
        {
            case "Low":
                return new EffectsDetailSettings("Simple", "Low", "Basic effects with minimal detail.");
            case "High":
                return new EffectsDetailSettings("Complex", "High", "High-quality effects with realistic detail.");
            case "Ultra":
                return new EffectsDetailSettings("Very Complex", "Very High", "Maximum effects detail with very realistic and dynamic visuals.");
            case "Medium":
            default:
                return new EffectsDetailSettings("Moderate", "Moderate", "Balanced effects with good detail.");
        }
    }

    // Property to get and set effects detail level
    public string EffectsDetail
    {
        get { return level; }
        set
        {
            level = value;
            effectsSettings = DetermineEffectsDetail(level);
            Debug.Log($"Effects Detail Level set to: {level}");
        }
    }

    // Adjust the effects detail based on system performance preference
    public void AdjustEffectsDetail(string performancePreference)
    {
        switch (performancePreference.ToLower())
        {
            case "low":
                EffectsDetail = "Low";
                break;
            case "medium":
                EffectsDetail = "Medium";
                break;
            case "high":
                EffectsDetail = "High";
                break;
            default:
                Debug.LogWarning("Unknown performance preference. Using default effects detail.");
                EffectsDetail = "Medium";
                break;
        }
    }

    // Display the current effects detail settings
    public void DisplayInfo()
    {
        Debug.Log($"Effects Detail Level: {level}");
        Debug.Log($"Complexity: {effectsSettings.complexity}");
        Debug.Log($"Resource Usage: {effectsSettings.resourceUsage}");
        Debug.Log($"Description: {effectsSettings.description}");
    }
}
