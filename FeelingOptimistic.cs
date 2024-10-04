using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingOptimistic : MonoBehaviour
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
        ApplyOptimisticFeeling();
    }

    public void ApplyOptimisticFeeling()
    {
        ApplyOptimisticToChakra(RootChakra);
        ApplyOptimisticToChakra(SacralChakra);
        ApplyOptimisticToChakra(SolarPlexusChakra);
        ApplyOptimisticToChakra(HeartChakra);
        ApplyOptimisticToChakra(ThroatChakra);
        ApplyOptimisticToChakra(ThirdEyeChakra);
        ApplyOptimisticToChakra(CrownChakra);
    }

    private void ApplyOptimisticToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Light, steady energy for optimism
            main.startSize = 0.5f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Continuous upward flow to reflect hope and positivity
        }
    }
}
