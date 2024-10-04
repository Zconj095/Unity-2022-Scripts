using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnergySpiralManager : MonoBehaviour
{

    [SerializeField] private ChakraType chakraType; // Chakra Type enum exposed to inspector
    [SerializeField] private GameObject chakraPrefab; // Prefab representing the chakra exposed to inspector
    [SerializeField] private ParticleSystem energySpiral; // Particle system for spiral effect exposed to inspector

    // Properties for configuring the spiral behavior
    [SerializeField] private float rotationSpeed = 50f; // Base speed of the spiral
    [SerializeField] private float spiralFrequency = 1f; // Frequency of the spiral waves
    [SerializeField] private float spiralRadius = 0.5f; // Radius of the spiral
    [SerializeField] private float energyStateMultiplier = 1f; // Multiplier for chakra state adjustment

    void Start()
    {
        // Initialize energy spiral based on the chakra type
        if (chakraPrefab != null)
        {
            if (energySpiral == null)
            {
                energySpiral = chakraPrefab.GetComponentInChildren<ParticleSystem>();
            }

            if (energySpiral != null)
            {
                ApplyChakraProperties();
            }
            else
            {
                Debug.LogError("Energy Spiral Particle System not found in the Chakra Prefab.");
            }
        }
    }

    private void ApplyChakraProperties()
    {
        var mainModule = energySpiral.main;
        var shape = energySpiral.shape;
        var colorOverLifetime = energySpiral.colorOverLifetime;

        Gradient gradient = new Gradient();

        // Configure color, speed, and other properties based on the specific chakra type
        float startSpeed = 0f;

        switch (chakraType)
        {
            case ChakraType.Root:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(Color.red, 0.0f), new GradientColorKey(Color.red, 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 7.83f; // Schumann resonance for root chakra
                shape.radius = spiralRadius;
                break;
            case ChakraType.Sacral:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(new Color(1.0f, 0.55f, 0.0f), 0.0f), new GradientColorKey(new Color(1.0f, 0.55f, 0.0f), 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 14.1f; // Schumann resonance for sacral chakra
                shape.radius = spiralRadius;
                break;
            case ChakraType.SolarPlexus:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(Color.yellow, 0.0f), new GradientColorKey(Color.yellow, 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 20.3f; // Schumann resonance for solar plexus chakra
                shape.radius = spiralRadius;
                break;
            case ChakraType.Heart:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(Color.green, 0.0f), new GradientColorKey(Color.green, 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 26.4f; // Schumann resonance for heart chakra
                shape.radius = spiralRadius;
                break;
            case ChakraType.Throat:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(Color.blue, 0.0f), new GradientColorKey(Color.blue, 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 33.8f; // Schumann resonance for throat chakra
                shape.radius = spiralRadius;
                break;
            case ChakraType.ThirdEye:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(new Color(0.29f, 0.0f, 0.51f), 0.0f), new GradientColorKey(new Color(0.29f, 0.0f, 0.51f), 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 39.2f; // Schumann resonance for third eye chakra
                shape.radius = spiralRadius;
                break;
            case ChakraType.Crown:
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(new Color(0.93f, 0.51f, 0.93f), 0.0f), new GradientColorKey(new Color(0.93f, 0.51f, 0.93f), 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(0.0f, 1.0f) }
                );
                startSpeed = 45.0f; // Schumann resonance for crown chakra
                shape.radius = spiralRadius;
                break;
        }

        // Apply color gradient
        colorOverLifetime.color = gradient;

        // Set properties for the spiral effect
        mainModule.startSpeed = startSpeed; // Assign the start speed
        mainModule.startLifetime = startSpeed / 10.0f; // Calculate lifetime based on speed

        var velocityOverLifetime = energySpiral.velocityOverLifetime;

        // Adjust orbital velocity for the spiral effect
        velocityOverLifetime.orbitalX = new ParticleSystem.MinMaxCurve(startSpeed * 0.1f);
        velocityOverLifetime.orbitalY = new ParticleSystem.MinMaxCurve(startSpeed * 0.05f);
    }
}

public enum ChakraType
{
    Root,
    Sacral,
    SolarPlexus,
    Heart,
    Throat,
    ThirdEye,
    Crown
}
