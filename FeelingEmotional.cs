using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingEmotional : MonoBehaviour
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
        ApplyEmotionalFeeling();
    }

    public void ApplyEmotionalFeeling()
    {
        ApplyEmotionalToChakra(RootChakra);
        ApplyEmotionalToChakra(SacralChakra);
        ApplyEmotionalToChakra(SolarPlexusChakra);
        ApplyEmotionalToChakra(HeartChakra);
        ApplyEmotionalToChakra(ThroatChakra);
        ApplyEmotionalToChakra(ThirdEyeChakra);
        ApplyEmotionalToChakra(CrownChakra);
    }

    private void ApplyEmotionalToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = Random.Range(0.7f, 2.0f); // Fluctuating speeds for emotional complexity
            main.startSize = Random.Range(0.4f, 0.7f); // Varied energy to reflect different emotional states
            main.startLifetime = Random.Range(2.5f, 4.0f);

            var emission = particleSystem.emission;
            emission.rateOverTime = Random.Range(20f, 45f); // Varied emission rates to reflect emotional fluctuations
        }
    }
}
