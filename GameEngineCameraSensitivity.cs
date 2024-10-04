using UnityEngine;

public class GameEngineCameraSensitivity : MonoBehaviour
{
    [SerializeField]
    private float sensitivity = 1.0f;  // Default camera sensitivity level

    // Property to get and set camera sensitivity
    public float CameraSensitivity
    {
        get { return sensitivity; }
        set
        {
            if (value <= 0)
            {
                Debug.LogError("Camera sensitivity must be a positive number.");
            }
            else
            {
                sensitivity = value;
                Debug.Log($"Camera Sensitivity set to: {sensitivity}");
            }
        }
    }

    // Method to adjust camera sensitivity based on user preference
    public void AdjustSensitivity(string preference)
    {
        switch (preference.ToLower())
        {
            case "low":
                CameraSensitivity = 0.5f;
                break;
            case "medium":
                CameraSensitivity = 1.0f;
                break;
            case "high":
                CameraSensitivity = 2.0f;
                break;
            default:
                Debug.LogWarning("Unknown preference. Using default sensitivity.");
                CameraSensitivity = 1.0f;
                break;
        }
    }

    // Method to display the current camera sensitivity setting
    public void DisplayInfo()
    {
        Debug.Log($"Camera Sensitivity: {sensitivity}");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
