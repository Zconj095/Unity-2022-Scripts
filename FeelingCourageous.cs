using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingCourageous : MonoBehaviour
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
        ApplyCourageousFeeling();
    }

    public void ApplyCourageousFeeling()
    {
        ApplyCourageousToChakra(RootChakra);
        ApplyCourageousToChakra(SacralChakra);
        ApplyCourageousToChakra(SolarPlexusChakra);
        ApplyCourageousToChakra(HeartChakra);
        ApplyCourageousToChakra(ThroatChakra);
        ApplyCourageousToChakra(ThirdEyeChakra);
        ApplyCourageousToChakra(CrownChakra);
    }

    private void ApplyCourageousToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.6f; // Fast and strong energy
            main.startSize = 0.6f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High energy output for bravery
        }
    }
}
