using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionSerious : MonoBehaviour
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
        ApplySeriousEmotion();
    }

    public void ApplySeriousEmotion()
    {
        ApplySeriousToChakra(RootChakra);
        ApplySeriousToChakra(SacralChakra);
        ApplySeriousToChakra(SolarPlexusChakra);
        ApplySeriousToChakra(HeartChakra);
        ApplySeriousToChakra(ThroatChakra);
        ApplySeriousToChakra(ThirdEyeChakra);
        ApplySeriousToChakra(CrownChakra);
    }

    private void ApplySeriousToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Steady, focused energy for seriousness
            main.startSize = 0.5f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Strong, focused flow representing discipline and concentration
        }
    }
}
