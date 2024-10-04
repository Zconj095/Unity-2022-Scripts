using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingImpressed : MonoBehaviour
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
        ApplyImpressedFeeling();
    }

    public void ApplyImpressedFeeling()
    {
        ApplyImpressedToChakra(RootChakra);
        ApplyImpressedToChakra(SacralChakra);
        ApplyImpressedToChakra(SolarPlexusChakra);
        ApplyImpressedToChakra(HeartChakra);
        ApplyImpressedToChakra(ThroatChakra);
        ApplyImpressedToChakra(ThirdEyeChakra);
        ApplyImpressedToChakra(CrownChakra);
    }

    private void ApplyImpressedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.4f; // Quick, reactive energy bursts for impressed state
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High energy flow reflecting satisfaction and reaction
        }
    }
}

