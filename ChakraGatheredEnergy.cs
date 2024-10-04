using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraGatheredEnergy : MonoBehaviour
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
        ApplyGatheredEnergyToChakras();
    }

    public void ApplyGatheredEnergyToChakras()
    {
        ApplyGatheredEnergyToChakra(RootChakra);
        ApplyGatheredEnergyToChakra(SacralChakra);
        ApplyGatheredEnergyToChakra(SolarPlexusChakra);
        ApplyGatheredEnergyToChakra(HeartChakra);
        ApplyGatheredEnergyToChakra(ThroatChakra);
        ApplyGatheredEnergyToChakra(ThirdEyeChakra);
        ApplyGatheredEnergyToChakra(CrownChakra);
    }

    private void ApplyGatheredEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Gathered energy flows slowly
            main.startSize = 1.2f; // Larger particles to represent accumulation
            main.startLifetime = 7.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 12f; // Gradual release of gathered energy
        }
    }
}
