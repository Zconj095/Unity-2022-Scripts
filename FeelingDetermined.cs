using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingDetermined : MonoBehaviour
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
        ApplyDeterminedFeeling();
    }

    public void ApplyDeterminedFeeling()
    {
        ApplyDeterminedToChakra(RootChakra);
        ApplyDeterminedToChakra(SacralChakra);
        ApplyDeterminedToChakra(SolarPlexusChakra);
        ApplyDeterminedToChakra(HeartChakra);
        ApplyDeterminedToChakra(ThroatChakra);
        ApplyDeterminedToChakra(ThirdEyeChakra);
        ApplyDeterminedToChakra(CrownChakra);
    }

    private void ApplyDeterminedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.8f; // Fast, focused energy for determination
            main.startSize = 0.7f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 45f; // High energy output for determination
        }
    }
}
