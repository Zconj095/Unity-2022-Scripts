using System.Collections.Generic;
using UnityEngine;
public class GameEngineCacheSunShadowMapsExampleUsage : MonoBehaviour
{
    private GameEngineCacheSunShadowMaps cacheSunShadowMapsSetting;

    void Start()
    {
        // Initialize the GameEngineCacheSunShadowMaps component
        cacheSunShadowMapsSetting = gameObject.AddComponent<GameEngineCacheSunShadowMaps>();

        // Display initial settings
        cacheSunShadowMapsSetting.DisplayInfo();

        // Generate shadow maps for regions
        cacheSunShadowMapsSetting.GenerateShadowMap("Forest");
        cacheSunShadowMapsSetting.GenerateShadowMap("Mountain");
        cacheSunShadowMapsSetting.DisplayInfo();

        // Use cached shadow map
        cacheSunShadowMapsSetting.GenerateShadowMap("Forest");

        // Disable caching and clear the cache
        cacheSunShadowMapsSetting.CacheSunShadowMaps = false;
        cacheSunShadowMapsSetting.DisplayInfo();
    }
}
