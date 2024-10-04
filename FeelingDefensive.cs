using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingDefensive : MonoBehaviour
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
        ApplyDefensiveFeeling();
    }

    public void ApplyDefensiveFeeling()
    {
        ApplyDefensiveToChakra(RootChakra);
        ApplyDefensiveToChakra(SacralChakra);
        ApplyDefensiveToChakra(SolarPlexusChakra);
        ApplyDefensiveToChakra(HeartChakra);
        ApplyDefensiveToChakra(ThroatChakra);
        ApplyDefensiveToChakra(ThirdEyeChakra);
        ApplyDefensiveToChakra(CrownChakra);
    }

    private void ApplyDefensiveToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Fast and constricted energy for defense
            main.startSize = 0.5f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Rapid emission for protective energy
        }
    }
}
