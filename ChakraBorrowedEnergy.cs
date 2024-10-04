using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChakraBorrowedEnergy : MonoBehaviour
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
        ApplyBorrowedEnergyToChakras();
    }

    public void ApplyBorrowedEnergyToChakras()
    {
        ApplyBorrowedEnergyToChakra(RootChakra);
        ApplyBorrowedEnergyToChakra(SacralChakra);
        ApplyBorrowedEnergyToChakra(SolarPlexusChakra);
        ApplyBorrowedEnergyToChakra(HeartChakra);
        ApplyBorrowedEnergyToChakra(ThroatChakra);
        ApplyBorrowedEnergyToChakra(ThirdEyeChakra);
        ApplyBorrowedEnergyToChakra(CrownChakra);
    }

    private void ApplyBorrowedEnergyToChakra(GameObject chakraLayer)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Borrowed energy moves quickly
            main.startSize = 0.8f;
            main.startLifetime = 5.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Fast, abundant emission
        }
    }
}
