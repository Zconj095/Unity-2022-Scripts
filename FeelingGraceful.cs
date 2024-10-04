using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingGraceful : MonoBehaviour
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
        ApplyGracefulFeeling();
    }

    public void ApplyGracefulFeeling()
    {
        ApplyGracefulToChakra(RootChakra);
        ApplyGracefulToChakra(SacralChakra);
        ApplyGracefulToChakra(SolarPlexusChakra);
        ApplyGracefulToChakra(HeartChakra);
        ApplyGracefulToChakra(ThroatChakra);
        ApplyGracefulToChakra(ThirdEyeChakra);
        ApplyGracefulToChakra(CrownChakra);
    }

    private void ApplyGracefulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Smooth, fluid energy
            main.startSize = 0.4f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Graceful flow for elegant movement
        }
    }
}
