using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodDefensive : MonoBehaviour
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
        ApplyDefensiveMood();
    }

    public void ApplyDefensiveMood()
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
            main.simulationSpeed = 1.2f; // More erratic and defensive behavior
            main.startSize = 0.4f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Denser particle emission for defensive state
        }
    }
}
