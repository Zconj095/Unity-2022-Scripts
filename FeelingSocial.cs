using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSocial : MonoBehaviour
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
        ApplySocialFeeling();
    }

    public void ApplySocialFeeling()
    {
        ApplySocialToChakra(RootChakra);
        ApplySocialToChakra(SacralChakra);
        ApplySocialToChakra(SolarPlexusChakra);
        ApplySocialToChakra(HeartChakra);
        ApplySocialToChakra(ThroatChakra);
        ApplySocialToChakra(ThirdEyeChakra);
        ApplySocialToChakra(CrownChakra);
    }

    private void ApplySocialToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Vibrant energy to reflect communication
            main.startSize = 0.5f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Continuous, outward flow representing interaction
        }
    }
}

