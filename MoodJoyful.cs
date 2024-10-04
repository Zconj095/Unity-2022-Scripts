using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodJoyful : MonoBehaviour
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
        ApplyJoyfulMood();
    }

    public void ApplyJoyfulMood()
    {
        ApplyJoyfulToChakra(RootChakra);
        ApplyJoyfulToChakra(SacralChakra);
        ApplyJoyfulToChakra(SolarPlexusChakra);
        ApplyJoyfulToChakra(HeartChakra);
        ApplyJoyfulToChakra(ThroatChakra);
        ApplyJoyfulToChakra(ThirdEyeChakra);
        ApplyJoyfulToChakra(CrownChakra);
    }

    private void ApplyJoyfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // High energy, joyful behavior
            main.startSize = 0.6f;
            main.startLifetime = 2f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Increased emission rate for joyful energy
        }
    }
}
