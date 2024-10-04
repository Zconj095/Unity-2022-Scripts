using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionSerene : MonoBehaviour
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
        ApplySereneEmotion();
    }

    public void ApplySereneEmotion()
    {
        ApplySereneToChakra(RootChakra);
        ApplySereneToChakra(SacralChakra);
        ApplySereneToChakra(SolarPlexusChakra);
        ApplySereneToChakra(HeartChakra);
        ApplySereneToChakra(ThroatChakra);
        ApplySereneToChakra(ThirdEyeChakra);
        ApplySereneToChakra(CrownChakra);
    }

    private void ApplySereneToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Smooth, calm energy reflecting serenity
            main.startSize = 0.5f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Even, flowing energy to reflect peacefulness and serenity
        }
    }
}
