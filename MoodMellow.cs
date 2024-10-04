using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodMellow : MonoBehaviour
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
        ApplyMellowMood();
    }

    public void ApplyMellowMood()
    {
        ApplyMellowToChakra(RootChakra);
        ApplyMellowToChakra(SacralChakra);
        ApplyMellowToChakra(SolarPlexusChakra);
        ApplyMellowToChakra(HeartChakra);
        ApplyMellowToChakra(ThroatChakra);
        ApplyMellowToChakra(ThirdEyeChakra);
        ApplyMellowToChakra(CrownChakra);
    }

    private void ApplyMellowToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.6f; // Slow, mellow behavior
            main.startSize = 0.3f;
            main.startLifetime = 4.0f; // Longer lifetime for a relaxed feel

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Lower emission rate for mellow energy
        }
    }
}
