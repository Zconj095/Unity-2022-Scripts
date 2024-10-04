using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraNaturalEnergy : MonoBehaviour
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
        ApplyNaturalEnergyToChakras();
    }

    public void ApplyNaturalEnergyToChakras()
    {
        ApplyNaturalEnergyToChakra(RootChakra);
        ApplyNaturalEnergyToChakra(SacralChakra);
        ApplyNaturalEnergyToChakra(SolarPlexusChakra);
        ApplyNaturalEnergyToChakra(HeartChakra);
        ApplyNaturalEnergyToChakra(ThroatChakra);
        ApplyNaturalEnergyToChakra(ThirdEyeChakra);
        ApplyNaturalEnergyToChakra(CrownChakra);
    }

    private void ApplyNaturalEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Natural energy for chakras flows dynamically
            main.startSize = 1.0f; // Larger particle size to represent balance
            main.startLifetime = 5.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Natural energy flows more frequently in chakras
        }
    }
}
