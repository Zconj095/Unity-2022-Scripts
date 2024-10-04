using UnityEngine;

public class GameEngineDisplayMode : MonoBehaviour
{
    [SerializeField]
    private string mode = "Fullscreen";  // Default display mode

    // Property to get and set display mode
    public string DisplayMode
    {
        get { return mode; }
        set
        {
            if (value != "Windowed" && value != "Fullscreen" && value != "Borderless Windowed")
            {
                Debug.LogError("Invalid display mode. Choose 'Windowed', 'Fullscreen', or 'Borderless Windowed'.");
            }
            else
            {
                mode = value;
                ApplyDisplayMode();
                Debug.Log($"Display Mode set to: {mode}");
            }
        }
    }

    // Toggle between windowed and fullscreen modes
    public void ToggleDisplayMode()
    {
        if (mode == "Fullscreen")
        {
            DisplayMode = "Windowed";
        }
        else if (mode == "Windowed")
        {
            DisplayMode = "Fullscreen";
        }
    }

    // Apply the current display mode setting
    private void ApplyDisplayMode()
    {
        switch (mode)
        {
            case "Windowed":
                Screen.fullScreenMode = FullScreenMode.Windowed;
                break;
            case "Fullscreen":
                Screen.fullScreenMode = FullScreenMode.ExclusiveFullScreen;
                break;
            case "Borderless Windowed":
                Screen.fullScreenMode = FullScreenMode.FullScreenWindow;
                break;
            default:
                Debug.LogWarning("Unknown display mode. Defaulting to Fullscreen.");
                Screen.fullScreenMode = FullScreenMode.ExclusiveFullScreen;
                break;
        }
    }

    // Display the current display mode setting
    public void DisplayInfo()
    {
        Debug.Log($"Display Mode: {mode}");
    }

    void Start()
    {
        ApplyDisplayMode();  // Apply the initial display mode setting
        DisplayInfo();  // Display initial settings
    }
}
