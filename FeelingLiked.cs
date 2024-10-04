using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingLiked : MonoBehaviour
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
        ApplyLikedFeeling();
    }

    public void ApplyLikedFeeling()
    {
        ApplyLikedToChakra(RootChakra);
        ApplyLikedToChakra(SacralChakra);
        ApplyLikedToChakra(SolarPlexusChakra);
        ApplyLikedToChakra(HeartChakra);
        ApplyLikedToChakra(ThroatChakra);
        ApplyLikedToChakra(ThirdEyeChakra);
        ApplyLikedToChakra(CrownChakra);
    }

    private void ApplyLikedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Gentle, open energy for acceptance and connection
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Calm and steady energy flow representing connection
        }
    }
}
