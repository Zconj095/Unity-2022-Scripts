using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingThrilled : MonoBehaviour
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
        ApplyThrilledFeeling();
    }

    public void ApplyThrilledFeeling()
    {
        ApplyThrilledToChakra(RootChakra);
        ApplyThrilledToChakra(SacralChakra);
        ApplyThrilledToChakra(SolarPlexusChakra);
        ApplyThrilledToChakra(HeartChakra);
        ApplyThrilledToChakra(ThroatChakra);
        ApplyThrilledToChakra(ThirdEyeChakra);
        ApplyThrilledToChakra(CrownChakra);
    }

    private void ApplyThrilledToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.8f; // Fast, intense energy for immense satisfaction
            main.startSize = 0.6f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // High energy output to reflect excitement and joy
        }
    }
}
