using System.Collections.Generic;
using UnityEngine;

public class GameEngineAmbientOcclusionExampleUsage : MonoBehaviour
{
    private GameEngineAmbientOcclusion aoSetting;

    void Start()
    {
        // Initialize the GameEngineAmbientOcclusion component
        aoSetting = gameObject.AddComponent<GameEngineAmbientOcclusion>();
        
        // Display initial settings
        aoSetting.DisplayInfo();

        // Adjust Ambient Occlusion settings to "High"
        aoSetting.AmbientOcclusionSettings = new GameEngineAmbientOcclusion.AOSettings("Pronounced", "High", "Strong AO effect with enhanced realism, higher resource usage.");
        aoSetting.DisplayInfo();

        // Adjust based on performance preference to "low"
        aoSetting.AdjustAO("low");
        aoSetting.DisplayInfo();
    }
}

