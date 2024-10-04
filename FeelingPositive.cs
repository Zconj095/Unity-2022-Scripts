using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingPositive : MonoBehaviour
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
        ApplyPositiveFeeling();
    }

    public void ApplyPositiveFeeling()
    {
        ApplyPositiveToChakra(RootChakra);
        ApplyPositiveToChakra(SacralChakra);
        ApplyPositiveToChakra(SolarPlexusChakra);
        ApplyPositiveToChakra(HeartChakra);
        ApplyPositiveToChakra(ThroatChakra);
        ApplyPositiveToChakra(ThirdEyeChakra);
        ApplyPositiveToChakra(CrownChakra);
    }

    private void ApplyPositiveToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Steady, strong energy for positivity
            main.startSize = 0.6f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // Strong, expansive energy flow reflecting positivity and higher values
        }
    }
}
