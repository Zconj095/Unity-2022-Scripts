using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingAcceptance : MonoBehaviour
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
        ApplyAcceptanceFeeling();
    }

    public void ApplyAcceptanceFeeling()
    {
        ApplyAcceptanceToChakra(RootChakra);
        ApplyAcceptanceToChakra(SacralChakra);
        ApplyAcceptanceToChakra(SolarPlexusChakra);
        ApplyAcceptanceToChakra(HeartChakra);
        ApplyAcceptanceToChakra(ThroatChakra);
        ApplyAcceptanceToChakra(ThirdEyeChakra);
        ApplyAcceptanceToChakra(CrownChakra);
    }

    private void ApplyAcceptanceToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Balanced energy flow for acceptance
            main.startSize = 0.4f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Moderate emission rate
        }
    }
}
