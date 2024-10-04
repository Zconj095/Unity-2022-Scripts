using System.Collections.Generic;
using UnityEngine;
public class AnisotropicFilteringExampleUsage : MonoBehaviour
{
    private GameEngineAnisotropicFiltering anisotropicFilteringSetting;
    private GameEngineScreenSpaceShadows screenSpaceShadowsSetting;

    void Start()
    {
        // Initialize and set up the Anisotropic Filtering setting
        anisotropicFilteringSetting = gameObject.AddComponent<GameEngineAnisotropicFiltering>();
        anisotropicFilteringSetting.AnisotropicFiltering = "16x";
        anisotropicFilteringSetting.DisplayInfo();

        // Initialize and set up the Screen Space Shadows setting
        screenSpaceShadowsSetting = gameObject.AddComponent<GameEngineScreenSpaceShadows>();
        screenSpaceShadowsSetting.ScreenSpaceShadows = (enabled: true, quality: "Medium");
        screenSpaceShadowsSetting.DisplayInfo();
    }
}
