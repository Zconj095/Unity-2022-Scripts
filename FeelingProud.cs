using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingProud : MonoBehaviour
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
        ApplyProudFeeling();
    }

    public void ApplyProudFeeling()
    {
        ApplyProudToChakra(RootChakra);
        ApplyProudToChakra(SacralChakra);
        ApplyProudToChakra(SolarPlexusChakra);
        ApplyProudToChakra(HeartChakra);
        ApplyProudToChakra(ThroatChakra);
        ApplyProudToChakra(ThirdEyeChakra);
        ApplyProudToChakra(CrownChakra);
    }

    private void ApplyProudToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Uplifting, steady energy for pride and accomplishment
            main.startSize = 0.6f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // Strong, controlled energy flow for achievement and success
        }
    }
}
