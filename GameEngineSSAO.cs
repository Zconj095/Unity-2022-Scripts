using UnityEngine;

public class GameEngineSSAO : MonoBehaviour
{
    [SerializeField]
    private bool enabled = true;  // Default SSAO enabled state

    [SerializeField]
    private string quality = "High";  // Default SSAO quality

    // Property to get and set SSAO settings
    public (bool enabled, string quality) SSAOSettings
    {
        get { return (enabled, quality); }
        set
        {
            enabled = value.enabled;
            quality = value.quality;
            if (quality != "Low" && quality != "Medium" && quality != "High")
            {
                Debug.LogError("Invalid quality setting. Choose 'Low', 'Medium', or 'High'.");
            }
            else
            {
                Debug.Log($"SSAO Settings updated: Enabled = {enabled}, Quality = {quality}");
            }
        }
    }

    // Method to display the current SSAO settings
    public void DisplayInfo()
    {
        string status = enabled ? "enabled" : "disabled";
        Debug.Log($"Screen Space Ambient Occlusion (SSAO): {status} ({quality} quality)");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
