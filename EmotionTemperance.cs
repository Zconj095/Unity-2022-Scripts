using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionTemperance : MonoBehaviour
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
        ApplyTemperanceEmotion();
    }

    public void ApplyTemperanceEmotion()
    {
        ApplyTemperanceToChakra(RootChakra);
        ApplyTemperanceToChakra(SacralChakra);
        ApplyTemperanceToChakra(SolarPlexusChakra);
        ApplyTemperanceToChakra(HeartChakra);
        ApplyTemperanceToChakra(ThroatChakra);
        ApplyTemperanceToChakra(ThirdEyeChakra);
        ApplyTemperanceToChakra(CrownChakra);
    }

    private void ApplyTemperanceToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Warm, gentle energy for temperance
            main.startSize = 0.5f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Warm, continuous flow representing peacefulness and balance
        }
    }
}
