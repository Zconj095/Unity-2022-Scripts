using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class MoodLoud : MonoBehaviour
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
        ApplyLoudMood();
    }

    public void ApplyLoudMood()
    {
        ApplyLoudToChakra(RootChakra);
        ApplyLoudToChakra(SacralChakra);
        ApplyLoudToChakra(SolarPlexusChakra);
        ApplyLoudToChakra(HeartChakra);
        ApplyLoudToChakra(ThroatChakra);
        ApplyLoudToChakra(ThirdEyeChakra);
        ApplyLoudToChakra(CrownChakra);
    }

    private void ApplyLoudToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 2.0f; // Erratic, fast energy
            main.startSize = 0.8f;
            main.startLifetime = 1.0f; // Shorter lifetime for sharp bursts

            var emission = particleSystem.emission;
            emission.rateOverTime = 60f; // High emission rate for loud behavior
        }
    }
}
