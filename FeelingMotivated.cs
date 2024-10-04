using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingMotivated : MonoBehaviour
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
        ApplyMotivatedFeeling();
    }

    public void ApplyMotivatedFeeling()
    {
        ApplyMotivatedToChakra(RootChakra);
        ApplyMotivatedToChakra(SacralChakra);
        ApplyMotivatedToChakra(SolarPlexusChakra);
        ApplyMotivatedToChakra(HeartChakra);
        ApplyMotivatedToChakra(ThroatChakra);
        ApplyMotivatedToChakra(ThirdEyeChakra);
        ApplyMotivatedToChakra(CrownChakra);
    }

    private void ApplyMotivatedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.8f; // Fast, intense energy for motivation
            main.startSize = 0.6f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 45f; // High energy output to reflect drive and passion
        }
    }
}
