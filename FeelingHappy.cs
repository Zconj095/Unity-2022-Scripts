using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingHappy : MonoBehaviour
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
        ApplyHappyFeeling();
    }

    public void ApplyHappyFeeling()
    {
        ApplyHappyToChakra(RootChakra);
        ApplyHappyToChakra(SacralChakra);
        ApplyHappyToChakra(SolarPlexusChakra);
        ApplyHappyToChakra(HeartChakra);
        ApplyHappyToChakra(ThroatChakra);
        ApplyHappyToChakra(ThirdEyeChakra);
        ApplyHappyToChakra(CrownChakra);
    }

    private void ApplyHappyToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.6f; // Fast, joyful energy for happiness
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High output for lively and positive energy
        }
    }
}
