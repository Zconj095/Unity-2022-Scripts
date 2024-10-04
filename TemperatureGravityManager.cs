using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public enum MoonPhase
{
    NewMoon,
    FullMoon,
    HalfMoon,
    CrescentMoon,
    CrescentWaxing,
    CrescentWaning,
    GibbousMoon,
    GibbousWaxing,
    GibbousWaning
}

public enum TimeOfDay
{
    Day,
    Night
}

public class TemperatureGravityManager : MonoBehaviour
{
    public Season season;
    public MoonPhase moonPhase;
    public TimeOfDay timeOfDay;

    public GameObject chakraPrefab;  // Prefab for chakras
    public GameObject auraPrefab;    // Prefab for auras

    private float temperature;
    private float gravity;

    private Dictionary<(Season, MoonPhase, TimeOfDay), (float temp, float grav)> environmentValues = new Dictionary<(Season, MoonPhase, TimeOfDay), (float temp, float grav)>();

    void Start()
    {
        AttachChakraPrefab(chakraPrefab);
        AttachAuraPrefab(auraPrefab);

        InitializeEnvironmentValues();
        ApplyEnvironmentalEffects();
        AdjustChakraAuraValues();
    }

    private void AttachChakraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Chakra Prefab attached for " + season + ", " + moonPhase + ", " + timeOfDay);
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
            Debug.Log("Aura Prefab attached for " + season + ", " + moonPhase + ", " + timeOfDay);
        }
        else
        {
            Debug.LogError("Aura Prefab is missing!");
        }
    }

    private void InitializeEnvironmentValues()
    {
        // Each tuple represents (Season, MoonPhase, TimeOfDay)
        // For each combination, define a temperature and gravity value
        environmentValues[(Season.Spring, MoonPhase.CrescentWaxing, TimeOfDay.Day)] = (18f, 9.8f);
        environmentValues[(Season.Spring, MoonPhase.CrescentWaning, TimeOfDay.Night)] = (14f, 9.78f);
        environmentValues[(Season.Summer, MoonPhase.FullMoon, TimeOfDay.Day)] = (32f, 9.85f);
        environmentValues[(Season.Autumn, MoonPhase.GibbousWaxing, TimeOfDay.Day)] = (16f, 9.76f);
        environmentValues[(Season.Winter, MoonPhase.GibbousWaning, TimeOfDay.Night)] = (-8f, 9.65f);
        // Continue defining for other combinations...

        environmentValues[(Season.Winter, MoonPhase.HalfMoon, TimeOfDay.Day)] = (-5f, 9.75f);
        environmentValues[(Season.Summer, MoonPhase.CrescentMoon, TimeOfDay.Night)] = (25f, 9.82f);
        environmentValues[(Season.Autumn, MoonPhase.GibbousMoon, TimeOfDay.Night)] = (10f, 9.77f);
        // Add more values for all possible combinations...
    }

    private void ApplyEnvironmentalEffects()
    {
        // Retrieve temperature and gravity based on the current season, moon phase, and time of day
        if (environmentValues.ContainsKey((season, moonPhase, timeOfDay)))
        {
            var values = environmentValues[(season, moonPhase, timeOfDay)];
            temperature = values.temp;
            gravity = values.grav;

            // Apply temperature and gravity settings (e.g., affecting the simulation)
            Physics.gravity = new Vector3(0, -gravity, 0);
            Debug.Log("Applied temperature: " + temperature + "°C and gravity: " + gravity + "m/s²");
        }
        else
        {
            Debug.LogError("No environmental data found for the current settings.");
        }
    }

    private void AdjustChakraAuraValues()
    {
        if (chakraPrefab != null)
        {
            ParticleSystem[] chakraParticleSystems = chakraPrefab.GetComponentsInChildren<ParticleSystem>();
            foreach (ParticleSystem ps in chakraParticleSystems)
            {
                var mainModule = ps.main;

                // Adjust particle behavior based on temperature, gravity, and time of day
                float timeOfDayMultiplier = (timeOfDay == TimeOfDay.Day) ? 1.0f : 0.85f; // Reduce slightly at night

                mainModule.simulationSpeed = Mathf.Lerp(0.5f, 2.0f, (temperature + 30) / 60) * timeOfDayMultiplier; // Slower if cold, faster if hot
                mainModule.startSize = Mathf.Lerp(0.8f, 1.5f, gravity / 9.81f) * timeOfDayMultiplier; // Adjust size based on gravity
                mainModule.startLifetime = Mathf.Lerp(4.0f, 8.0f, temperature / 30.0f) * timeOfDayMultiplier; // Adjust lifetime based on temperature
            }
        }

        if (auraPrefab != null)
        {
            ParticleSystem[] auraParticleSystems = auraPrefab.GetComponentsInChildren<ParticleSystem>();
            foreach (ParticleSystem ps in auraParticleSystems)
            {
                var mainModule = ps.main;

                // Similar adjustments for Aura
                float timeOfDayMultiplier = (timeOfDay == TimeOfDay.Day) ? 1.0f : 0.85f; // Reduce slightly at night

                mainModule.simulationSpeed = Mathf.Lerp(0.5f, 2.0f, (temperature + 30) / 60) * timeOfDayMultiplier;
                mainModule.startSize = Mathf.Lerp(0.8f, 1.5f, gravity / 9.81f) * timeOfDayMultiplier;
                mainModule.startLifetime = Mathf.Lerp(4.0f, 8.0f, temperature / 30.0f) * timeOfDayMultiplier;
            }
        }
    }
}
