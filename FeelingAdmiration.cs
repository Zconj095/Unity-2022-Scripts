using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingAdmiration : MonoBehaviour
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
        ApplyAdmirationFeeling();
    }

    public void ApplyAdmirationFeeling()
    {
        ApplyAdmirationToChakra(RootChakra);
        ApplyAdmirationToChakra(SacralChakra);
        ApplyAdmirationToChakra(SolarPlexusChakra);
        ApplyAdmirationToChakra(HeartChakra);
        ApplyAdmirationToChakra(ThroatChakra);
        ApplyAdmirationToChakra(ThirdEyeChakra);
        ApplyAdmirationToChakra(CrownChakra);
    }

    private void ApplyAdmirationToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.1f; // Controlled, focused energy for admiration
            main.startSize = 0.6f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Steady and dignified emission
        }
    }
}
