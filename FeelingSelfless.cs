using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSelfless : MonoBehaviour
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
        ApplySelflessFeeling();
    }

    public void ApplySelflessFeeling()
    {
        ApplySelflessToChakra(RootChakra);
        ApplySelflessToChakra(SacralChakra);
        ApplySelflessToChakra(SolarPlexusChakra);
        ApplySelflessToChakra(HeartChakra);
        ApplySelflessToChakra(ThroatChakra);
        ApplySelflessToChakra(ThirdEyeChakra);
        ApplySelflessToChakra(CrownChakra);
    }

    private void ApplySelflessToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.1f; // Smooth, continuous energy outward
            main.startSize = 0.5f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Steady outward flow to represent selflessness
        }
    }
}
