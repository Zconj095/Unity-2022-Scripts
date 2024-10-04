using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionLove : MonoBehaviour
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
        ApplyLoveEmotion();
    }

    public void ApplyLoveEmotion()
    {
        ApplyLoveToChakra(RootChakra);
        ApplyLoveToChakra(SacralChakra);
        ApplyLoveToChakra(SolarPlexusChakra);
        ApplyLoveToChakra(HeartChakra);
        ApplyLoveToChakra(ThroatChakra);
        ApplyLoveToChakra(ThirdEyeChakra);
        ApplyLoveToChakra(CrownChakra);
    }

    private void ApplyLoveToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Slow, gentle energy reflecting love
            main.startSize = 0.6f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Gentle, continuous flow to reflect the encompassing nature of love
        }
    }
}
