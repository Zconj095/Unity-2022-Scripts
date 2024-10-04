using System.Collections.Generic;
using UnityEngine;

public class GameEngineCacheSunShadowMaps : MonoBehaviour
{
    [SerializeField]
    private bool enabled = true;  // Default state for caching sun shadow maps

    private Dictionary<string, string> cachedShadowMaps = new Dictionary<string, string>();

    // Property to get and set the cache sun shadow maps setting
    public bool CacheSunShadowMaps
    {
        get { return enabled; }
        set
        {
            enabled = value;
            if (!enabled)
            {
                cachedShadowMaps.Clear();  // Clear the cache if caching is disabled
                Debug.Log("Sun shadow map caching disabled, cache cleared.");
            }
            else
            {
                Debug.Log("Sun shadow map caching enabled.");
            }
        }
    }

    // Method to simulate the generation of a shadow map for a specific region
    public string GenerateShadowMap(string region)
    {
        if (enabled && cachedShadowMaps.ContainsKey(region))
        {
            Debug.Log($"Using cached shadow map for region: {region}");
            return cachedShadowMaps[region];
        }
        else
        {
            Debug.Log($"Generating new shadow map for region: {region}");
            string shadowMap = $"ShadowMap_{region}";
            if (enabled)
            {
                cachedShadowMaps[region] = shadowMap;
                Debug.Log($"Cached shadow map for region: {region}");
            }
            return shadowMap;
        }
    }

    // Method to display the current cache sun shadow maps setting
    public void DisplayInfo()
    {
        string status = enabled ? "enabled" : "disabled";
        Debug.Log($"Cache Sun Shadow Maps: {status}");
        Debug.Log($"Cached Shadow Maps: {(cachedShadowMaps.Count > 0 ? string.Join(", ", cachedShadowMaps.Keys) : "None")}");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
