using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionHope : MonoBehaviour
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
        ApplyHopeEmotion();
    }

    public void ApplyHopeEmotion()
    {
        ApplyHopeToChakra(RootChakra);
        ApplyHopeToChakra(SacralChakra);
        ApplyHopeToChakra(SolarPlexusChakra);
        ApplyHopeToChakra(HeartChakra);
        ApplyHopeToChakra(ThroatChakra);
        ApplyHopeToChakra(ThirdEyeChakra);
        ApplyHopeToChakra(CrownChakra);
    }

    private void ApplyHopeToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Gradual upward flow for hope
            main.startSize = 0.5f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Continuous, upward flow to represent hope and perseverance
        }
    }
}
