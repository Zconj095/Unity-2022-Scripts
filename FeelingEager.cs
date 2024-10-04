using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingEager : MonoBehaviour
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
        ApplyEagerFeeling();
    }

    public void ApplyEagerFeeling()
    {
        ApplyEagerToChakra(RootChakra);
        ApplyEagerToChakra(SacralChakra);
        ApplyEagerToChakra(SolarPlexusChakra);
        ApplyEagerToChakra(HeartChakra);
        ApplyEagerToChakra(ThroatChakra);
        ApplyEagerToChakra(ThirdEyeChakra);
        ApplyEagerToChakra(CrownChakra);
    }

    private void ApplyEagerToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Fast energy for anticipation
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // High energy output to reflect eagerness
        }
    }
}
