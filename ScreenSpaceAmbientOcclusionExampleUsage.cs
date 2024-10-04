using UnityEngine;

public class ScreenSpaceAmbientOcclusionExampleUsage : MonoBehaviour
{
    private GameEngineSSAO ssaoSetting;

    void Start()
    {
        // Initialize the GameEngineSSAO component
        ssaoSetting = gameObject.AddComponent<GameEngineSSAO>();

        // Display initial settings
        ssaoSetting.DisplayInfo();

        // Update SSAO settings to medium quality
        ssaoSetting.SSAOSettings = (enabled: true, quality: "Medium");
        ssaoSetting.DisplayInfo();
    }
}

public class ScreenSpaceAmbientOcclusionSettingExampleUsage : MonoBehaviour
{
    private GameEngineSSAO ssaoSetting;

    void Start()
    {
        // Initialize the GameEngineSSAO component
        ssaoSetting = gameObject.AddComponent<GameEngineSSAO>();

        // Display initial settings
        ssaoSetting.DisplayInfo();

        // Update SSAO settings to medium quality
        ssaoSetting.SSAOSettings = (enabled: true, quality: "Medium");
        ssaoSetting.DisplayInfo();
    }
}
