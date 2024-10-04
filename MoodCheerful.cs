using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodCheerful : MonoBehaviour
{
    // Attach existing chakra GameObjects here in the Unity inspector
    public GameObject RootChakra;
    public GameObject SacralChakra;
    public GameObject SolarPlexusChakra;
    public GameObject HeartChakra;
    public GameObject ThroatChakra;
    public GameObject ThirdEyeChakra;
    public GameObject CrownChakra;

    void Start()
    {
        ApplyCheerfulMood();
    }

    public void ApplyCheerfulMood()
    {
        ApplyCheerfulToChakra(RootChakra);
        ApplyCheerfulToChakra(SacralChakra);
        ApplyCheerfulToChakra(SolarPlexusChakra);
        ApplyCheerfulToChakra(HeartChakra);
        ApplyCheerfulToChakra(ThroatChakra);
        ApplyCheerfulToChakra(ThirdEyeChakra);
        ApplyCheerfulToChakra(CrownChakra);
    }

    private void ApplyCheerfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Cheerful effect: speed up particles
            main.startSize = 0.5f;
            main.startLifetime = 2f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Cheerful emission rate for all chakras
        }
    }
}
