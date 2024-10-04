using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraEconomicalEnergy : MonoBehaviour
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
        ApplyEconomicalEnergyToChakras();
    }

    public void ApplyEconomicalEnergyToChakras()
    {
        ApplyEconomicalEnergyToChakra(RootChakra);
        ApplyEconomicalEnergyToChakra(SacralChakra);
        ApplyEconomicalEnergyToChakra(SolarPlexusChakra);
        ApplyEconomicalEnergyToChakra(HeartChakra);
        ApplyEconomicalEnergyToChakra(ThroatChakra);
        ApplyEconomicalEnergyToChakra(ThirdEyeChakra);
        ApplyEconomicalEnergyToChakra(CrownChakra);
    }

    private void ApplyEconomicalEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.7f; // Economical energy flows slowly
            main.startSize = 0.5f;
            main.startLifetime = 9.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 8f; // Sparse emission due to conservation
        }
    }
}
