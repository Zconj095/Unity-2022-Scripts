using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingBrave : MonoBehaviour
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
        ApplyBraveFeeling();
    }

    public void ApplyBraveFeeling()
    {
        ApplyBraveToChakra(RootChakra);
        ApplyBraveToChakra(SacralChakra);
        ApplyBraveToChakra(SolarPlexusChakra);
        ApplyBraveToChakra(HeartChakra);
        ApplyBraveToChakra(ThroatChakra);
        ApplyBraveToChakra(ThirdEyeChakra);
        ApplyBraveToChakra(CrownChakra);
    }

    private void ApplyBraveToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.8f; // Fast, intense energy for bravery
            main.startSize = 0.7f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // Strong, fast energy to reflect courage
        }
    }
}

