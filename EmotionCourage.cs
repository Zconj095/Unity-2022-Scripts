using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionCourage : MonoBehaviour
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
        ApplyCourageEmotion();
    }

    public void ApplyCourageEmotion()
    {
        ApplyCourageToChakra(RootChakra);
        ApplyCourageToChakra(SacralChakra);
        ApplyCourageToChakra(SolarPlexusChakra);
        ApplyCourageToChakra(HeartChakra);
        ApplyCourageToChakra(ThroatChakra);
        ApplyCourageToChakra(ThirdEyeChakra);
        ApplyCourageToChakra(CrownChakra);
    }

    private void ApplyCourageToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.7f; // Strong, forward-moving energy for courage
            main.startSize = 0.6f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 45f; // High energy output reflecting bravery and strength
        }
    }
}
