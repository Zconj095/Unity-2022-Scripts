using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingNeutral : MonoBehaviour
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
        ApplyNeutralFeeling();
    }

    public void ApplyNeutralFeeling()
    {
        ApplyNeutralToChakra(RootChakra);
        ApplyNeutralToChakra(SacralChakra);
        ApplyNeutralToChakra(SolarPlexusChakra);
        ApplyNeutralToChakra(HeartChakra);
        ApplyNeutralToChakra(ThroatChakra);
        ApplyNeutralToChakra(ThirdEyeChakra);
        ApplyNeutralToChakra(CrownChakra);
    }

    private void ApplyNeutralToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Calm, even energy for neutrality
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Balanced, steady flow without extremes
        }
    }
}
