using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraObtainedEnergy : MonoBehaviour
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
        ApplyObtainedEnergyToChakras();
    }

    public void ApplyObtainedEnergyToChakras()
    {
        ApplyObtainedEnergyToChakra(RootChakra);
        ApplyObtainedEnergyToChakra(SacralChakra);
        ApplyObtainedEnergyToChakra(SolarPlexusChakra);
        ApplyObtainedEnergyToChakra(HeartChakra);
        ApplyObtainedEnergyToChakra(ThroatChakra);
        ApplyObtainedEnergyToChakra(ThirdEyeChakra);
        ApplyObtainedEnergyToChakra(CrownChakra);
    }

    private void ApplyObtainedEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Obtained energy moves faster than natural
            main.startSize = 0.9f;
            main.startLifetime = 4.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Moderate flow of obtained energy
        }
    }
}
