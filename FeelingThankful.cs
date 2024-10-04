using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingThankful : MonoBehaviour
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
        ApplyThankfulFeeling();
    }

    public void ApplyThankfulFeeling()
    {
        ApplyThankfulToChakra(RootChakra);
        ApplyThankfulToChakra(SacralChakra);
        ApplyThankfulToChakra(SolarPlexusChakra);
        ApplyThankfulToChakra(HeartChakra);
        ApplyThankfulToChakra(ThroatChakra);
        ApplyThankfulToChakra(ThirdEyeChakra);
        ApplyThankfulToChakra(CrownChakra);
    }

    private void ApplyThankfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Smooth, calm energy for gratitude and relief
            main.startSize = 0.4f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Gentle, steady flow for thankfulness
        }
    }
}
