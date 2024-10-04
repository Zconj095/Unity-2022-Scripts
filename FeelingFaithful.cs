using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingFaithful : MonoBehaviour
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
        ApplyFaithfulFeeling();
    }

    public void ApplyFaithfulFeeling()
    {
        ApplyFaithfulToChakra(RootChakra);
        ApplyFaithfulToChakra(SacralChakra);
        ApplyFaithfulToChakra(SolarPlexusChakra);
        ApplyFaithfulToChakra(HeartChakra);
        ApplyFaithfulToChakra(ThroatChakra);
        ApplyFaithfulToChakra(ThirdEyeChakra);
        ApplyFaithfulToChakra(CrownChakra);
    }

    private void ApplyFaithfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Constant, unwavering energy
            main.startSize = 0.6f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Steady and strong flow to reflect loyalty
        }
    }
}
