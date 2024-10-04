using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingPowerful : MonoBehaviour
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
        ApplyPowerfulFeeling();
    }

    public void ApplyPowerfulFeeling()
    {
        ApplyPowerfulToChakra(RootChakra);
        ApplyPowerfulToChakra(SacralChakra);
        ApplyPowerfulToChakra(SolarPlexusChakra);
        ApplyPowerfulToChakra(HeartChakra);
        ApplyPowerfulToChakra(ThroatChakra);
        ApplyPowerfulToChakra(ThirdEyeChakra);
        ApplyPowerfulToChakra(CrownChakra);
    }

    private void ApplyPowerfulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.7f; // Intense, steady energy for power
            main.startSize = 0.7f;
            main.startLifetime = 3.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // High energy output reflecting capability and power
        }
    }
}
