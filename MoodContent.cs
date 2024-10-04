using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodContent : MonoBehaviour
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
        ApplyContentMood();
    }

    public void ApplyContentMood()
    {
        ApplyContentToChakra(RootChakra);
        ApplyContentToChakra(SacralChakra);
        ApplyContentToChakra(SolarPlexusChakra);
        ApplyContentToChakra(HeartChakra);
        ApplyContentToChakra(ThroatChakra);
        ApplyContentToChakra(ThirdEyeChakra);
        ApplyContentToChakra(CrownChakra);
    }

    private void ApplyContentToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Relaxed, consistent energy flow
            main.startSize = 0.3f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 12f; // Gentle emission rate for contentment
        }
    }
}

