using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingCollected : MonoBehaviour
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
        ApplyCollectedFeeling();
    }

    public void ApplyCollectedFeeling()
    {
        ApplyCollectedToChakra(RootChakra);
        ApplyCollectedToChakra(SacralChakra);
        ApplyCollectedToChakra(SolarPlexusChakra);
        ApplyCollectedToChakra(HeartChakra);
        ApplyCollectedToChakra(ThroatChakra);
        ApplyCollectedToChakra(ThirdEyeChakra);
        ApplyCollectedToChakra(CrownChakra);
    }

    private void ApplyCollectedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Steady and focused energy for collected state
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Balanced emission for calm, focused energy
        }
    }
}
