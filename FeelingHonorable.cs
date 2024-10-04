using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingHonorable : MonoBehaviour
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
        ApplyHonorableFeeling();
    }

    public void ApplyHonorableFeeling()
    {
        ApplyHonorableToChakra(RootChakra);
        ApplyHonorableToChakra(SacralChakra);
        ApplyHonorableToChakra(SolarPlexusChakra);
        ApplyHonorableToChakra(HeartChakra);
        ApplyHonorableToChakra(ThroatChakra);
        ApplyHonorableToChakra(ThirdEyeChakra);
        ApplyHonorableToChakra(CrownChakra);
    }

    private void ApplyHonorableToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Balanced and steady energy for commitment
            main.startSize = 0.5f;
            main.startLifetime = 3.2f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 28f; // Strong, controlled flow reflecting honor in fulfilling promises
        }
    }
}
