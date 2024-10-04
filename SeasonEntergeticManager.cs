using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Season
{
    Spring,
    Summer,
    Autumn,
    Winter
}


public class SeasonEnergeticManager : MonoBehaviour
{
    public Season season;  // For seasonal changes
    public GameObject chakraPrefab; // Prefab for chakras
    public GameObject auraPrefab; // Prefab for auras

    private Dictionary<Season, (float speed, float size, float lifetime, float emissionRate)> chakraSeasonParticleBehaviors = new Dictionary<Season, (float speed, float size, float lifetime, float emissionRate)>();
    private Dictionary<Season, (float speed, float size, float lifetime, float emissionRate)> auraSeasonParticleBehaviors = new Dictionary<Season, (float speed, float size, float lifetime, float emissionRate)>();

    void Start()
    {
        // Assign chakra and aura prefabs for the selected season
        AttachChakraPrefab(chakraPrefab);
        AttachAuraPrefab(auraPrefab);

        // Set different energetic reactions based on the season
        InitializeSeasonalParticleBehaviors();

        ApplyChakraParticleBehavior();
        ApplyAuraParticleBehavior();
    }

    private void AttachChakraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Chakra Prefab attached for " + season.ToString());
        }
        else
        {
            Debug.LogError("Chakra Prefab is missing!");
        }
    }

    private void AttachAuraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Aura Prefab attached for " + season.ToString());
        }
        else
        {
            Debug.LogError("Aura Prefab is missing!");
        }
    }

    private void InitializeSeasonalParticleBehaviors()
    {
        // Define seasonal particle behaviors for chakras
        chakraSeasonParticleBehaviors[Season.Spring] = (1.2f, 1.1f, 5.0f, 30f);
        chakraSeasonParticleBehaviors[Season.Summer] = (1.5f, 1.3f, 4.5f, 40f);
        chakraSeasonParticleBehaviors[Season.Autumn] = (1.0f, 1.0f, 6.0f, 25f);
        chakraSeasonParticleBehaviors[Season.Winter] = (0.7f, 1.2f, 7.0f, 20f);

        // Define seasonal particle behaviors for auras
        auraSeasonParticleBehaviors[Season.Spring] = (1.3f, 1.2f, 5.0f, 35f);
        auraSeasonParticleBehaviors[Season.Summer] = (1.7f, 1.5f, 4.0f, 50f);
        auraSeasonParticleBehaviors[Season.Autumn] = (1.0f, 1.0f, 6.5f, 30f);
        auraSeasonParticleBehaviors[Season.Winter] = (0.6f, 1.2f, 7.5f, 15f);
    }

    private void ApplyChakraParticleBehavior()
    {
        if (chakraPrefab != null && chakraSeasonParticleBehaviors.ContainsKey(season))
        {
            ParticleSystem[] chakraParticleSystems = chakraPrefab.GetComponentsInChildren<ParticleSystem>();
            foreach (ParticleSystem ps in chakraParticleSystems)
            {
                var mainModule = ps.main;

                // Apply seasonal values for Chakra
                mainModule.simulationSpeed = chakraSeasonParticleBehaviors[season].speed;
                mainModule.startSize = chakraSeasonParticleBehaviors[season].size;
                mainModule.startLifetime = chakraSeasonParticleBehaviors[season].lifetime;

                var emissionModule = ps.emission;
                emissionModule.rateOverTime = chakraSeasonParticleBehaviors[season].emissionRate;
            }
            Debug.Log("Applied Chakra Particle Behavior for " + season);
        }
        else
        {
            Debug.LogError("No chakra particle system found or behavior missing for " + season);
        }
    }

    private void ApplyAuraParticleBehavior()
    {
        if (auraPrefab != null && auraSeasonParticleBehaviors.ContainsKey(season))
        {
            ParticleSystem[] auraParticleSystems = auraPrefab.GetComponentsInChildren<ParticleSystem>();
            foreach (ParticleSystem ps in auraParticleSystems)
            {
                var mainModule = ps.main;

                // Apply seasonal values for Aura
                mainModule.simulationSpeed = auraSeasonParticleBehaviors[season].speed;
                mainModule.startSize = auraSeasonParticleBehaviors[season].size;
                mainModule.startLifetime = auraSeasonParticleBehaviors[season].lifetime;

                var emissionModule = ps.emission;
                emissionModule.rateOverTime = auraSeasonParticleBehaviors[season].emissionRate;
            }
            Debug.Log("Applied Aura Particle Behavior for " + season);
        }
        else
        {
            Debug.LogError("No aura particle system found or behavior missing for " + season);
        }
    }
}
