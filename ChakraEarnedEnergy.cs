using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraEarnedEnergy : MonoBehaviour
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
        ApplyEarnedEnergyToChakras();
    }

    public void ApplyEarnedEnergyToChakras()
    {
        ApplyEarnedEnergyToChakra(RootChakra);
        ApplyEarnedEnergyToChakra(SacralChakra);
        ApplyEarnedEnergyToChakra(SolarPlexusChakra);
        ApplyEarnedEnergyToChakra(HeartChakra);
        ApplyEarnedEnergyToChakra(ThroatChakra);
        ApplyEarnedEnergyToChakra(ThirdEyeChakra);
        ApplyEarnedEnergyToChakra(CrownChakra);
    }

    private void ApplyEarnedEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Balanced, stable flow
            main.startSize = 0.9f;
            main.startLifetime = 6.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Steady flow reflecting earned energy
        }
    }
}
