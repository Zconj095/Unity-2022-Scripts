using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingFaith : MonoBehaviour
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
        ApplyFaithFeeling();
    }

    public void ApplyFaithFeeling()
    {
        ApplyFaithToChakra(RootChakra);
        ApplyFaithToChakra(SacralChakra);
        ApplyFaithToChakra(SolarPlexusChakra);
        ApplyFaithToChakra(HeartChakra);
        ApplyFaithToChakra(ThroatChakra);
        ApplyFaithToChakra(ThirdEyeChakra);
        ApplyFaithToChakra(CrownChakra);
    }

    private void ApplyFaithToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Steady, purposeful energy
            main.startSize = 0.5f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Continuous energy to reflect belief and desire
        }
    }
}
