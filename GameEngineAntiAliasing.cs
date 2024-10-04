using UnityEngine;

public class GameEngineAntiAliasing : MonoBehaviour
{
    // Make the anti-aliasing method and quality visible in the Unity Editor
    [SerializeField]
    private string method = "TAA";  // Default antialiasing method

    [SerializeField]
    private string quality = "High";  // Default quality level

    // Property to get and set antialiasing settings
    public (string method, string quality) AntialiasingSettings
    {
        get { return (method, quality); }
        set
        {
            method = value.method;
            quality = value.quality;
            Debug.Log($"Antialiasing Settings updated: Method = {method}, Quality = {quality}");
        }
    }

    // Adjust antialiasing based on system performance
    public void AdjustAntialiasing(string performanceLevel)
    {
        switch (performanceLevel.ToLower())
        {
            case "low":
                method = "FXAA";
                quality = "Low";
                break;
            case "medium":
                method = "TAA";
                quality = "Medium";
                break;
            case "high":
                method = "MSAA";
                quality = "High";
                break;
            default:
                Debug.LogWarning("Unknown performance level. Using default settings.");
                method = "TAA";
                quality = "High";
                break;
        }
        Debug.Log($"Antialiasing adjusted for {performanceLevel} performance: Method = {method}, Quality = {quality}");
    }

    // Display the current antialiasing settings
    public void DisplayInfo()
    {
        Debug.Log($"Antialiasing Method: {method}");
        Debug.Log($"Antialiasing Quality: {quality}");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
