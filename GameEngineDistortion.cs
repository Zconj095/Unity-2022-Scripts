using UnityEngine;

public class GameEngineDistortion : MonoBehaviour
{
    [SerializeField]
    private bool enabled = true;  // Default distortion enabled state

    [SerializeField, Range(0.0f, 2.0f)]
    private float intensity = 1.0f;  // Default distortion intensity

    // Property to get and set distortion settings
    public (bool enabled, float intensity) DistortionSettings
    {
        get { return (enabled, intensity); }
        set
        {
            enabled = value.enabled;
            intensity = Mathf.Clamp(value.intensity, 0.0f, 2.0f);
            Debug.Log($"Distortion Settings updated: Enabled = {enabled}, Intensity = {intensity}");
        }
    }

    // Adjust the distortion effect based on system performance preference
    public void AdjustDistortion(string performancePreference)
    {
        switch (performancePreference.ToLower())
        {
            case "low":
                intensity = 0.5f;
                break;
            case "medium":
                intensity = 1.0f;
                break;
            case "high":
                intensity = 1.5f;
                break;
            default:
                Debug.LogWarning("Unknown performance preference. Using default intensity.");
                intensity = 1.0f;
                break;
        }
        Debug.Log($"Distortion adjusted for {performancePreference} performance: Intensity = {intensity}");
    }

    // Display the current distortion settings
    public void DisplayInfo()
    {
        string status = enabled ? "enabled" : "disabled";
        Debug.Log($"Distortion Effect: {status} (Intensity: {intensity})");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
