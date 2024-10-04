using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodEnergetic : MonoBehaviour
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
        ApplyEnergeticMood();
    }

    public void ApplyEnergeticMood()
    {
        ApplyEnergeticToChakra(RootChakra);
        ApplyEnergeticToChakra(SacralChakra);
        ApplyEnergeticToChakra(SolarPlexusChakra);
        ApplyEnergeticToChakra(HeartChakra);
        ApplyEnergeticToChakra(ThroatChakra);
        ApplyEnergeticToChakra(ThirdEyeChakra);
        ApplyEnergeticToChakra(CrownChakra);
    }

    private void ApplyEnergeticToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 2.0f; // Fast, energetic behavior
            main.startSize = 0.6f;
            main.startLifetime = 1.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // High-energy particle emission
        }
    }
}
