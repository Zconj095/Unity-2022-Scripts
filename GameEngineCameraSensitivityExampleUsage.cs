using UnityEngine;

public class GameEngineCameraSensitivityExampleUsage : MonoBehaviour
{
    private GameEngineCameraSensitivity cameraSensitivitySetting;

    void Start()
    {
        // Initialize the GameEngineCameraSensitivity component
        cameraSensitivitySetting = gameObject.AddComponent<GameEngineCameraSensitivity>();

        // Display initial settings
        cameraSensitivitySetting.DisplayInfo();

        // Update camera sensitivity
        cameraSensitivitySetting.CameraSensitivity = 1.5f;
        cameraSensitivitySetting.DisplayInfo();

        // Adjust sensitivity based on preference
        cameraSensitivitySetting.AdjustSensitivity("low");
        cameraSensitivitySetting.DisplayInfo();
    }
}
