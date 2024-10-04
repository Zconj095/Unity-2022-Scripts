using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingFascinated : MonoBehaviour
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
        ApplyFascinatedFeeling();
    }

    public void ApplyFascinatedFeeling()
    {
        ApplyFascinatedToChakra(RootChakra);
        ApplyFascinatedToChakra(SacralChakra);
        ApplyFascinatedToChakra(SolarPlexusChakra);
        ApplyFascinatedToChakra(HeartChakra);
        ApplyFascinatedToChakra(ThroatChakra);
        ApplyFascinatedToChakra(ThirdEyeChakra);
        ApplyFascinatedToChakra(CrownChakra);
    }

    private void ApplyFascinatedToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.6f; // Lively, fast energy for fascination
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High output to reflect excitement and satisfaction
        }
    }
}
