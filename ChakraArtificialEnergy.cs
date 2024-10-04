using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraArtificialEnergy : MonoBehaviour
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
        ApplyArtificialEnergyToChakras();
    }

    public void ApplyArtificialEnergyToChakras()
    {
        ApplyArtificialEnergyToChakra(RootChakra);
        ApplyArtificialEnergyToChakra(SacralChakra);
        ApplyArtificialEnergyToChakra(SolarPlexusChakra);
        ApplyArtificialEnergyToChakra(HeartChakra);
        ApplyArtificialEnergyToChakra(ThroatChakra);
        ApplyArtificialEnergyToChakra(ThirdEyeChakra);
        ApplyArtificialEnergyToChakra(CrownChakra);
    }

    private void ApplyArtificialEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Faster flow for artificial energy
            main.startSize = 0.6f; // Smaller particles to reflect artificial nature
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // Higher emission due to stimulation of artificial energy
        }
    }
}
