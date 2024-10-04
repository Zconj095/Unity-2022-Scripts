using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionHappiness : MonoBehaviour
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
        ApplyHappinessEmotion();
    }

    public void ApplyHappinessEmotion()
    {
        ApplyHappinessToChakra(RootChakra);
        ApplyHappinessToChakra(SacralChakra);
        ApplyHappinessToChakra(SolarPlexusChakra);
        ApplyHappinessToChakra(HeartChakra);
        ApplyHappinessToChakra(ThroatChakra);
        ApplyHappinessToChakra(ThirdEyeChakra);
        ApplyHappinessToChakra(CrownChakra);
    }

    private void ApplyHappinessToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.6f; // Fast, bright energy for happiness
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High energy output reflecting joy and happiness
        }
    }
}
