using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingThoughtful : MonoBehaviour
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
        ApplyThoughtfulFeeling();
    }

    public void ApplyThoughtfulFeeling()
    {
        ApplyThoughtfulToChakra(RootChakra);
        ApplyThoughtfulToChakra(SacralChakra);
        ApplyThoughtfulToChakra(SolarPlexusChakra);
        ApplyThoughtfulToChakra(HeartChakra);
        ApplyThoughtfulToChakra(ThroatChakra);
        ApplyThoughtfulToChakra(ThirdEyeChakra);
        ApplyThoughtfulToChakra(CrownChakra);
    }

    private void ApplyThoughtfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Slow, deliberate energy reflecting thoughtfulness
            main.startSize = 0.5f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Calm, steady energy for positive intentions
        }
    }
}
