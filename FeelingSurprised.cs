using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSurprised : MonoBehaviour
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
        ApplySurprisedFeeling();
    }

    public void ApplySurprisedFeeling()
    {
        ApplySurprisedToChakra(RootChakra);
        ApplySurprisedToChakra(SacralChakra);
        ApplySurprisedToChakra(SolarPlexusChakra);
        ApplySurprisedToChakra(HeartChakra);
        ApplySurprisedToChakra(ThroatChakra);
        ApplySurprisedToChakra(ThirdEyeChakra);
        ApplySurprisedToChakra(CrownChakra);
    }

    private void ApplySurprisedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.8f; // Quick bursts of energy for surprise
            main.startSize = 0.5f;
            main.startLifetime = 2.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // Sudden bursts to reflect surprise followed by steady energy
        }
    }
}
