using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingHonor : MonoBehaviour
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
        ApplyHonorFeeling();
    }

    public void ApplyHonorFeeling()
    {
        ApplyHonorToChakra(RootChakra);
        ApplyHonorToChakra(SacralChakra);
        ApplyHonorToChakra(SolarPlexusChakra);
        ApplyHonorToChakra(HeartChakra);
        ApplyHonorToChakra(ThroatChakra);
        ApplyHonorToChakra(ThirdEyeChakra);
        ApplyHonorToChakra(CrownChakra);
    }

    private void ApplyHonorToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Steady and strong energy for honor
            main.startSize = 0.6f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Solid and focused energy flow to reflect duty and purpose
        }
    }
}
