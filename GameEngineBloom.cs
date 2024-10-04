using UnityEngine;

public class GameEngineBloom : MonoBehaviour
{
    // These fields will now be visible in the Unity Editor
    [SerializeField]
    private bool enabled = true;  // Default state of the bloom effect

    [SerializeField, Range(0.0f, 1.0f)]
    private float intensity = 0.5f;  // Default intensity of the bloom effect

    // Property to get and set bloom settings
    public (bool enabled, float intensity) BloomSettings
    {
        get { return (enabled, intensity); }
        set
        {
            enabled = value.enabled;
            intensity = Mathf.Clamp(value.intensity, 0.0f, 1.0f);
            Debug.Log($"Bloom Settings updated: Enabled = {enabled}, Intensity = {intensity}");
        }
    }

    // Adjust bloom intensity based on user preference
    public void AdjustBloom(string preference)
    {
        switch (preference.ToLower())
        {
            case "low":
                intensity = 0.2f;
                break;
            case "medium":
                intensity = 0.5f;
                break;
            case "high":
                intensity = 0.8f;
                break;
            default:
                Debug.LogWarning("Unknown preference. Using default intensity.");
                intensity = 0.5f;
                break;
        }
        Debug.Log($"Bloom intensity adjusted based on {preference} preference: Intensity = {intensity}");
    }

    // Display the current bloom settings
    public void DisplayInfo()
    {
        string status = enabled ? "enabled" : "disabled";
        Debug.Log($"Bloom Effect: {status} (Intensity: {intensity})");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
