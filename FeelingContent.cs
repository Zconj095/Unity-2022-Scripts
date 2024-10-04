using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingContent : MonoBehaviour
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
        ApplyContentFeeling();
    }

    public void ApplyContentFeeling()
    {
        ApplyContentToChakra(RootChakra);
        ApplyContentToChakra(SacralChakra);
        ApplyContentToChakra(SolarPlexusChakra);
        ApplyContentToChakra(HeartChakra);
        ApplyContentToChakra(ThroatChakra);
        ApplyContentToChakra(ThirdEyeChakra);
        ApplyContentToChakra(CrownChakra);
    }

    private void ApplyContentToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Calm and smooth energy
            main.startSize = 0.4f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Steady flow to reflect contentment
        }
    }
}
