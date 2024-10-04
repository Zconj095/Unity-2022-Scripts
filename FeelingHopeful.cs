using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingHopeful : MonoBehaviour
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
        ApplyHopefulFeeling();
    }

    public void ApplyHopefulFeeling()
    {
        ApplyHopefulToChakra(RootChakra);
        ApplyHopefulToChakra(SacralChakra);
        ApplyHopefulToChakra(SolarPlexusChakra);
        ApplyHopefulToChakra(HeartChakra);
        ApplyHopefulToChakra(ThroatChakra);
        ApplyHopefulToChakra(ThirdEyeChakra);
        ApplyHopefulToChakra(CrownChakra);
    }

    private void ApplyHopefulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.1f; // Light, upward energy for hope
            main.startSize = 0.5f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 32f; // Continuous, ascending energy reflecting hope and optimism
        }
    }
}
