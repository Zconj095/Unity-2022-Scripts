using UnityEngine;

public class GameEngineDeepLearningSuperSampling : MonoBehaviour
{
    [SerializeField]
    private bool enabled = true;  // Default DLSS enabled state

    [SerializeField]
    private string quality = "Balanced";  // Default DLSS quality

    private DLSSSettings settings;

    // Class to represent DLSS settings
    private class DLSSSettings
    {
        public string description;
        public float upscaleFactor;

        public DLSSSettings(string description, float upscaleFactor)
        {
            this.description = description;
            this.upscaleFactor = upscaleFactor;
        }
    }

    void Start()
    {
        settings = DetermineDLSSSettings(quality);
        DisplayInfo();  // Display initial settings
    }

    // Determine the DLSS settings based on the selected quality level
    private DLSSSettings DetermineDLSSSettings(string quality)
    {
        switch (quality)
        {
            case "Performance":
                return new DLSSSettings("Maximized performance, lower image quality.", 2.0f);
            case "Quality":
                return new DLSSSettings("Maximized image quality, less performance gain.", 1.25f);
            case "Balanced":
            default:
                return new DLSSSettings("Balanced between performance and image quality.", 1.5f);
        }
    }

    // Property to get and set DLSS settings
    public string DLSSQuality
    {
        get { return quality; }
        set
        {
            quality = value;
            settings = DetermineDLSSSettings(quality);
            Debug.Log($"DLSS Settings updated: Quality = {quality}, Description = {settings.description}, Upscale Factor = {settings.upscaleFactor}");
        }
    }

    // Adjust the DLSS effect based on system performance preference
    public void AdjustDLSS(string performancePreference)
    {
        switch (performancePreference.ToLower())
        {
            case "low":
                DLSSQuality = "Performance";
                break;
            case "medium":
                DLSSQuality = "Balanced";
                break;
            case "high":
                DLSSQuality = "Quality";
                break;
            default:
                Debug.LogWarning("Unknown performance preference. Using default quality setting.");
                DLSSQuality = "Balanced";
                break;
        }
    }

    // Display the current DLSS settings
    public void DisplayInfo()
    {
        string status = enabled ? "enabled" : "disabled";
        Debug.Log($"Deep Learning Super Sampling (DLSS): {status}");
        Debug.Log($"Quality: {quality}");
        Debug.Log($"Description: {settings.description}");
        Debug.Log($"Upscale Factor: {settings.upscaleFactor}");
    }
}
