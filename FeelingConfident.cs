using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingConfident : MonoBehaviour
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
        ApplyConfidentFeeling();
    }

    public void ApplyConfidentFeeling()
    {
        ApplyConfidentToChakra(RootChakra);
        ApplyConfidentToChakra(SacralChakra);
        ApplyConfidentToChakra(SolarPlexusChakra);
        ApplyConfidentToChakra(HeartChakra);
        ApplyConfidentToChakra(ThroatChakra);
        ApplyConfidentToChakra(ThirdEyeChakra);
        ApplyConfidentToChakra(CrownChakra);
    }

    private void ApplyConfidentToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Strong, steady energy for confidence
            main.startSize = 0.6f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Solid and unwavering energy flow
        }
    }
}
