using UnityEngine;

public class GameEngineDisplayResolution : MonoBehaviour
{
    [SerializeField]
    private int width = 1920;  // Default horizontal resolution in pixels

    [SerializeField]
    private int height = 1080;  // Default vertical resolution in pixels

    // Property to get and set the resolution
    public (int width, int height) Resolution
    {
        get { return (width, height); }
        set
        {
            width = value.width;
            height = value.height;
            ApplyResolution();
            Debug.Log($"Resolution set to: {width}x{height}");
        }
    }

    // Method to adjust the resolution by a scaling factor
    public void AdjustResolution(float scalingFactor)
    {
        width = Mathf.RoundToInt(width * scalingFactor);
        height = Mathf.RoundToInt(height * scalingFactor);
        ApplyResolution();
        Debug.Log($"Resolution adjusted to: {width}x{height}");
    }

    // Apply the current resolution
    private void ApplyResolution()
    {
        Screen.SetResolution(width, height, Screen.fullScreenMode);
    }

    // Display the current resolution and related information
    public void DisplayInfo()
    {
        Debug.Log($"Current Resolution: {width}x{height}");
        Debug.Log($"Total Pixels: {width * height}");
    }

    void Start()
    {
        ApplyResolution();  // Apply the initial resolution setting
        DisplayInfo();  // Display initial settings
    }
}
