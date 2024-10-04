using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingRelaxed : MonoBehaviour
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
        ApplyRelaxedFeeling();
    }

    public void ApplyRelaxedFeeling()
    {
        ApplyRelaxedToChakra(RootChakra);
        ApplyRelaxedToChakra(SacralChakra);
        ApplyRelaxedToChakra(SolarPlexusChakra);
        ApplyRelaxedToChakra(HeartChakra);
        ApplyRelaxedToChakra(ThroatChakra);
        ApplyRelaxedToChakra(ThirdEyeChakra);
        ApplyRelaxedToChakra(CrownChakra);
    }

    private void ApplyRelaxedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Slow, smooth energy for relaxation
            main.startSize = 0.4f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Gentle, continuous energy flow to reflect contentment and ease
        }
    }
}
