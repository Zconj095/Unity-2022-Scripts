using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodQuiet : MonoBehaviour
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
        ApplyQuietMood();
    }

    public void ApplyQuietMood()
    {
        ApplyQuietToChakra(RootChakra);
        ApplyQuietToChakra(SacralChakra);
        ApplyQuietToChakra(SolarPlexusChakra);
        ApplyQuietToChakra(HeartChakra);
        ApplyQuietToChakra(ThroatChakra);
        ApplyQuietToChakra(ThirdEyeChakra);
        ApplyQuietToChakra(CrownChakra);
    }

    private void ApplyQuietToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.5f; // Very slow energy for quiet behavior
            main.startSize = 0.2f;
            main.startLifetime = 5.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 8f; // Very low emission rate for quietness
        }
    }
}
