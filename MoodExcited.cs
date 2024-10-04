using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodExcited : MonoBehaviour
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
        ApplyExcitedMood();
    }

    public void ApplyExcitedMood()
    {
        ApplyExcitedToChakra(RootChakra);
        ApplyExcitedToChakra(SacralChakra);
        ApplyExcitedToChakra(SolarPlexusChakra);
        ApplyExcitedToChakra(HeartChakra);
        ApplyExcitedToChakra(ThroatChakra);
        ApplyExcitedToChakra(ThirdEyeChakra);
        ApplyExcitedToChakra(CrownChakra);
    }

    private void ApplyExcitedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.8f; // Rapid and excited energy
            main.startSize = 0.4f;
            main.startLifetime = 2.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // Higher frequency for excitement
        }
    }
}

