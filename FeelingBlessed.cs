using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingBlessed : MonoBehaviour
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
        ApplyBlessedFeeling();
    }

    public void ApplyBlessedFeeling()
    {
        ApplyBlessedToChakra(RootChakra);
        ApplyBlessedToChakra(SacralChakra);
        ApplyBlessedToChakra(SolarPlexusChakra);
        ApplyBlessedToChakra(HeartChakra);
        ApplyBlessedToChakra(ThroatChakra);
        ApplyBlessedToChakra(ThirdEyeChakra);
        ApplyBlessedToChakra(CrownChakra);
    }

    private void ApplyBlessedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.1f; // Smooth yet strong energy for blessedness
            main.startSize = 0.6f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Expansive, enriched energy
        }
    }
}
