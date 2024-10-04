using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraWilledEnergy : MonoBehaviour
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
        ApplyWilledEnergyToChakras();
    }

    public void ApplyWilledEnergyToChakras()
    {
        ApplyWilledEnergyToChakra(RootChakra);
        ApplyWilledEnergyToChakra(SacralChakra);
        ApplyWilledEnergyToChakra(SolarPlexusChakra);
        ApplyWilledEnergyToChakra(HeartChakra);
        ApplyWilledEnergyToChakra(ThroatChakra);
        ApplyWilledEnergyToChakra(ThirdEyeChakra);
        ApplyWilledEnergyToChakra(CrownChakra);
    }

    private void ApplyWilledEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Controlled flow for Willed energy
            main.startSize = 0.8f;
            main.startLifetime = 6.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Steady flow reflecting control
        }
    }
}
