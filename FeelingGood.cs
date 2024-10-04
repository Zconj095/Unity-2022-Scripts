using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingGood : MonoBehaviour
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
        ApplyGoodFeeling();
    }

    public void ApplyGoodFeeling()
    {
        ApplyGoodToChakra(RootChakra);
        ApplyGoodToChakra(SacralChakra);
        ApplyGoodToChakra(SolarPlexusChakra);
        ApplyGoodToChakra(HeartChakra);
        ApplyGoodToChakra(ThroatChakra);
        ApplyGoodToChakra(ThirdEyeChakra);
        ApplyGoodToChakra(CrownChakra);
    }

    private void ApplyGoodToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Smooth, balanced energy
            main.startSize = 0.5f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Gentle and positive energy flow
        }
    }
}
