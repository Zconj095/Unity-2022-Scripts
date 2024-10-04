using System.Collections.Generic;
using UnityEngine;
public class GameEngineBloomExampleUsage : MonoBehaviour
{
    private GameEngineBloom bloomSetting;

    void Start()
    {
        // Initialize the GameEngineBloom component
        bloomSetting = gameObject.AddComponent<GameEngineBloom>();

        // Display initial settings
        bloomSetting.DisplayInfo();

        // Update bloom settings to enable with high intensity
        bloomSetting.BloomSettings = (enabled: true, intensity: 0.8f);
        bloomSetting.DisplayInfo();

        // Adjust bloom for low-intensity preference
        bloomSetting.AdjustBloom("low");
        bloomSetting.DisplayInfo();
    }
}
