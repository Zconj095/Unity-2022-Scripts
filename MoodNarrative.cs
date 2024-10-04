using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodNarrative : MonoBehaviour
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
        ApplyNarrativeMood();
    }

    public void ApplyNarrativeMood()
    {
        ApplyNarrativeToChakra(RootChakra);
        ApplyNarrativeToChakra(SacralChakra);
        ApplyNarrativeToChakra(SolarPlexusChakra);
        ApplyNarrativeToChakra(HeartChakra);
        ApplyNarrativeToChakra(ThroatChakra);
        ApplyNarrativeToChakra(ThirdEyeChakra);
        ApplyNarrativeToChakra(CrownChakra);
    }

    private void ApplyNarrativeToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Rhythmic and flowing energy
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Moderate emission rate to reflect emphasis
        }
    }
}
