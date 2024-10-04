using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraStoredEnergy : MonoBehaviour
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
        ApplyStoredEnergyToChakras();
    }

    public void ApplyStoredEnergyToChakras()
    {
        ApplyStoredEnergyToChakra(RootChakra);
        ApplyStoredEnergyToChakra(SacralChakra);
        ApplyStoredEnergyToChakra(SolarPlexusChakra);
        ApplyStoredEnergyToChakra(HeartChakra);
        ApplyStoredEnergyToChakra(ThroatChakra);
        ApplyStoredEnergyToChakra(ThirdEyeChakra);
        ApplyStoredEnergyToChakra(CrownChakra);
    }

    private void ApplyStoredEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.5f; // Stored energy moves slowly
            main.startSize = 0.7f;
            main.startLifetime = 8.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 10f; // Slower, steadier release
        }
    }
}
