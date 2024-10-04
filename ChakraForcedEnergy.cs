using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraForcedEnergy : MonoBehaviour
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
        ApplyForcedEnergyToChakras();
    }

    public void ApplyForcedEnergyToChakras()
    {
        ApplyForcedEnergyToChakra(RootChakra);
        ApplyForcedEnergyToChakra(SacralChakra);
        ApplyForcedEnergyToChakra(SolarPlexusChakra);
        ApplyForcedEnergyToChakra(HeartChakra);
        ApplyForcedEnergyToChakra(ThroatChakra);
        ApplyForcedEnergyToChakra(ThirdEyeChakra);
        ApplyForcedEnergyToChakra(CrownChakra);
    }

    private void ApplyForcedEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 2.0f; // Forced energy moves faster and more erratically
            main.startSize = 1.2f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // High frequency reflecting force
        }
    }
}
