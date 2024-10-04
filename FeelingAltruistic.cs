using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingAltruistic : MonoBehaviour
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
        ApplyAltruisticFeeling();
    }

    public void ApplyAltruisticFeeling()
    {
        ApplyAltruisticToChakra(RootChakra);
        ApplyAltruisticToChakra(SacralChakra);
        ApplyAltruisticToChakra(SolarPlexusChakra);
        ApplyAltruisticToChakra(HeartChakra);
        ApplyAltruisticToChakra(ThroatChakra);
        ApplyAltruisticToChakra(ThirdEyeChakra);
        ApplyAltruisticToChakra(CrownChakra);
    }

    private void ApplyAltruisticToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Steady flow for a sense of purpose
            main.startSize = 0.5f;
            main.startLifetime = 3.0f; // Longer lifetime to represent lasting belief

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Consistent, unified energy flow
        }
    }
}
