using UnityEngine;

public class GameEngineAnisotropicFiltering : MonoBehaviour
{
    // Make the anisotropic filtering level visible in the Unity Editor
    [SerializeField]
    private string level = "4x";  // Default anisotropic filtering level

    // Property to get and set anisotropic filtering settings
    public string AnisotropicFiltering
    {
        get { return level; }
        set
        {
            if (value == "2x" || value == "4x" || value == "8x" || value == "16x")
            {
                level = value;
                Debug.Log($"Anisotropic Filtering Level set to: {level}");
            }
            else
            {
                Debug.LogError("Invalid filtering level. Choose '2x', '4x', '8x', or '16x'.");
            }
        }
    }

    public void DisplayInfo()
    {
        Debug.Log($"Anisotropic Filtering Level: {level}");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
