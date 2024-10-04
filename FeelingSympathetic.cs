using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSympathetic : MonoBehaviour
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
        ApplySympatheticFeeling();
    }

    public void ApplySympatheticFeeling()
    {
        ApplySympatheticToChakra(RootChakra);
        ApplySympatheticToChakra(SacralChakra);
        ApplySympatheticToChakra(SolarPlexusChakra);
        ApplySympatheticToChakra(HeartChakra);
        ApplySympatheticToChakra(ThroatChakra);
        ApplySympatheticToChakra(ThirdEyeChakra);
        ApplySympatheticToChakra(CrownChakra);
    }

    private void ApplySympatheticToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Calm, gentle energy for empathy and understanding
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Steady outward flow reflecting compassion and connection
        }
    }
}
