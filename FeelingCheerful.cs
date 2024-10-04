using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingCheerful : MonoBehaviour
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
        ApplyCheerfulFeeling();
    }

    public void ApplyCheerfulFeeling()
    {
        ApplyCheerfulToChakra(RootChakra);
        ApplyCheerfulToChakra(SacralChakra);
        ApplyCheerfulToChakra(SolarPlexusChakra);
        ApplyCheerfulToChakra(HeartChakra);
        ApplyCheerfulToChakra(ThroatChakra);
        ApplyCheerfulToChakra(ThirdEyeChakra);
        ApplyCheerfulToChakra(CrownChakra);
    }

    private void ApplyCheerfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Fast and lively energy for cheerfulness
            main.startSize = 0.5f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // High frequency for cheerful energy
        }
    }
}
