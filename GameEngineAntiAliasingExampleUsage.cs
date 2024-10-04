using UnityEngine;

public class GameEngineAntiAliasingExampleUsage : MonoBehaviour
{
    private GameEngineAntiAliasing aaSetting;

    void Start()
    {
        // Initialize the GameEngineAntiAliasing component
        aaSetting = gameObject.AddComponent<GameEngineAntiAliasing>();

        // Display initial settings
        aaSetting.DisplayInfo();

        // Update antialiasing settings to MSAA with Ultra quality
        aaSetting.AntialiasingSettings = ("MSAA", "Ultra");
        aaSetting.DisplayInfo();

        // Adjust antialiasing for low-performance systems
        aaSetting.AdjustAntialiasing("low");
        aaSetting.DisplayInfo();
    }
}
