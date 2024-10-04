using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingIndependent : MonoBehaviour
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
        ApplyIndependentFeeling();
    }

    public void ApplyIndependentFeeling()
    {
        ApplyIndependentToChakra(RootChakra);
        ApplyIndependentToChakra(SacralChakra);
        ApplyIndependentToChakra(SolarPlexusChakra);
        ApplyIndependentToChakra(HeartChakra);
        ApplyIndependentToChakra(ThroatChakra);
        ApplyIndependentToChakra(ThirdEyeChakra);
        ApplyIndependentToChakra(CrownChakra);
    }

    private void ApplyIndependentToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Steady, strong energy for independence
            main.startSize = 0.6f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Direct, self-sustained energy flow for independence
        }
    }
}
