using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionSerenity : MonoBehaviour
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
        ApplySerenityEmotion();
    }

    public void ApplySerenityEmotion()
    {
        ApplySerenityToChakra(RootChakra);
        ApplySerenityToChakra(SacralChakra);
        ApplySerenityToChakra(SolarPlexusChakra);
        ApplySerenityToChakra(HeartChakra);
        ApplySerenityToChakra(ThroatChakra);
        ApplySerenityToChakra(ThirdEyeChakra);
        ApplySerenityToChakra(CrownChakra);
    }

    private void ApplySerenityToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Slow, smooth energy for deep serenity
            main.startSize = 0.5f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Gentle, flowing energy to reflect serenity and calm satisfaction
        }
    }
}
