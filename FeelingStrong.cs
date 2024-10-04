using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingStrong : MonoBehaviour
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
        ApplyStrongFeeling();
    }

    public void ApplyStrongFeeling()
    {
        ApplyStrongToChakra(RootChakra);
        ApplyStrongToChakra(SacralChakra);
        ApplyStrongToChakra(SolarPlexusChakra);
        ApplyStrongToChakra(HeartChakra);
        ApplyStrongToChakra(ThroatChakra);
        ApplyStrongToChakra(ThirdEyeChakra);
        ApplyStrongToChakra(CrownChakra);
    }

    private void ApplyStrongToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Steady, strong energy flow for strength and willpower
            main.startSize = 0.6f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 45f; // High energy output to reflect resilience and determination
        }
    }
}
