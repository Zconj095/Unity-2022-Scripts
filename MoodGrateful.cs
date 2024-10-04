using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodGrateful : MonoBehaviour
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
        ApplyGratefulMood();
    }

    public void ApplyGratefulMood()
    {
        ApplyGratefulToChakra(RootChakra);
        ApplyGratefulToChakra(SacralChakra);
        ApplyGratefulToChakra(SolarPlexusChakra);
        ApplyGratefulToChakra(HeartChakra);
        ApplyGratefulToChakra(ThroatChakra);
        ApplyGratefulToChakra(ThirdEyeChakra);
        ApplyGratefulToChakra(CrownChakra);
    }

    private void ApplyGratefulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Calm but uplifting energy
            main.startSize = 0.3f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Steady and peaceful for gratitude
        }
    }
}
