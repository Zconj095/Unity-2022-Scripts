using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingFocused : MonoBehaviour
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
        ApplyFocusedFeeling();
    }

    public void ApplyFocusedFeeling()
    {
        ApplyFocusedToChakra(RootChakra);
        ApplyFocusedToChakra(SacralChakra);
        ApplyFocusedToChakra(SolarPlexusChakra);
        ApplyFocusedToChakra(HeartChakra);
        ApplyFocusedToChakra(ThroatChakra);
        ApplyFocusedToChakra(ThirdEyeChakra);
        ApplyFocusedToChakra(CrownChakra);
    }

    private void ApplyFocusedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Controlled, directed energy for focus
            main.startSize = 0.4f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Steady, concentrated energy output
        }
    }
}
