using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodRelaxed : MonoBehaviour
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
        ApplyRelaxedMood();
    }

    public void ApplyRelaxedMood()
    {
        ApplyRelaxedToChakra(RootChakra);
        ApplyRelaxedToChakra(SacralChakra);
        ApplyRelaxedToChakra(SolarPlexusChakra);
        ApplyRelaxedToChakra(HeartChakra);
        ApplyRelaxedToChakra(ThroatChakra);
        ApplyRelaxedToChakra(ThirdEyeChakra);
        ApplyRelaxedToChakra(CrownChakra);
    }

    private void ApplyRelaxedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Relaxed, steady flow of energy
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 12f; // Steady, relaxed emission
        }
    }
}
