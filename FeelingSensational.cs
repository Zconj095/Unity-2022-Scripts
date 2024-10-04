using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSensational : MonoBehaviour
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
        ApplySensationalFeeling();
    }

    public void ApplySensationalFeeling()
    {
        ApplySensationalToChakra(RootChakra);
        ApplySensationalToChakra(SacralChakra);
        ApplySensationalToChakra(SolarPlexusChakra);
        ApplySensationalToChakra(HeartChakra);
        ApplySensationalToChakra(ThroatChakra);
        ApplySensationalToChakra(ThirdEyeChakra);
        ApplySensationalToChakra(CrownChakra);
    }

    private void ApplySensationalToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.6f; // Dynamic energy for sensory experiences
            main.startSize = Random.Range(0.4f, 0.7f); // Fluctuating energy sizes
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High, varied energy flow to reflect sensory enjoyment
        }
    }
}
