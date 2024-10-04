using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingConcerned : MonoBehaviour
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
        ApplyConcernedFeeling();
    }

    public void ApplyConcernedFeeling()
    {
        ApplyConcernedToChakra(RootChakra);
        ApplyConcernedToChakra(SacralChakra);
        ApplyConcernedToChakra(SolarPlexusChakra);
        ApplyConcernedToChakra(HeartChakra);
        ApplyConcernedToChakra(ThroatChakra);
        ApplyConcernedToChakra(ThirdEyeChakra);
        ApplyConcernedToChakra(CrownChakra);
    }

    private void ApplyConcernedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.1f; // Slightly faster to reflect tension
            main.startSize = 0.3f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Steady but slightly erratic to reflect worry
        }
    }
}
