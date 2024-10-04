using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingFullOfPurpose : MonoBehaviour
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
        ApplyFullOfPurposeFeeling();
    }

    public void ApplyFullOfPurposeFeeling()
    {
        ApplyFullOfPurposeToChakra(RootChakra);
        ApplyFullOfPurposeToChakra(SacralChakra);
        ApplyFullOfPurposeToChakra(SolarPlexusChakra);
        ApplyFullOfPurposeToChakra(HeartChakra);
        ApplyFullOfPurposeToChakra(ThroatChakra);
        ApplyFullOfPurposeToChakra(ThirdEyeChakra);
        ApplyFullOfPurposeToChakra(CrownChakra);
    }

    private void ApplyFullOfPurposeToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Steady, deliberate energy for purpose
            main.startSize = 0.6f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Strong and meaningful energy flow
        }
    }
}
