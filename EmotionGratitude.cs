using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionGratitude : MonoBehaviour
{
    public GameObject RootChakra;
    public GameObject SacralChakra;
    public GameObject SolarPlexusChakra;
    public GameObject HeartChakra;
    public GameObject ThroatChakra;
    public GameObject ThirdEyeChakra;
    public GameObject CrownChakra;

    void Start()
    {
        ApplyGratitudeEmotion();
    }

    public void ApplyGratitudeEmotion()
    {
        ApplyGratitudeToChakra(RootChakra);
        ApplyGratitudeToChakra(SacralChakra);
        ApplyGratitudeToChakra(SolarPlexusChakra);
        ApplyGratitudeToChakra(HeartChakra);
        ApplyGratitudeToChakra(ThroatChakra);
        ApplyGratitudeToChakra(ThirdEyeChakra);
        ApplyGratitudeToChakra(CrownChakra);
    }

    private void ApplyGratitudeToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Smooth, calm energy for gratitude
            main.startSize = 0.4f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Gentle, flowing energy for appreciation and thankfulness
        }
    }
}
