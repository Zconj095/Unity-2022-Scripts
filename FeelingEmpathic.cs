using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingEmpathic : MonoBehaviour
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
        ApplyEmpathicFeeling();
    }

    public void ApplyEmpathicFeeling()
    {
        ApplyEmpathicToChakra(RootChakra);
        ApplyEmpathicToChakra(SacralChakra);
        ApplyEmpathicToChakra(SolarPlexusChakra);
        ApplyEmpathicToChakra(HeartChakra);
        ApplyEmpathicToChakra(ThroatChakra);
        ApplyEmpathicToChakra(ThirdEyeChakra);
        ApplyEmpathicToChakra(CrownChakra);
    }

    private void ApplyEmpathicToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Steady energy for deep emotional connection
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Consistent emission to reflect calm empathy
        }
    }
}
