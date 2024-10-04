using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingLoved : MonoBehaviour
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
        ApplyLovedFeeling();
    }

    public void ApplyLovedFeeling()
    {
        ApplyLovedToChakra(RootChakra);
        ApplyLovedToChakra(SacralChakra);
        ApplyLovedToChakra(SolarPlexusChakra);
        ApplyLovedToChakra(HeartChakra);
        ApplyLovedToChakra(ThroatChakra);
        ApplyLovedToChakra(ThirdEyeChakra);
        ApplyLovedToChakra(CrownChakra);
    }

    private void ApplyLovedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Soft, smooth energy for love and serenity
            main.startSize = 0.5f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Gentle, steady energy flow to reflect safety and comfort
        }
    }
}
