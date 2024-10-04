using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodPeaceful : MonoBehaviour
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
        ApplyPeacefulMood();
    }

    public void ApplyPeacefulMood()
    {
        ApplyPeacefulToChakra(RootChakra);
        ApplyPeacefulToChakra(SacralChakra);
        ApplyPeacefulToChakra(SolarPlexusChakra);
        ApplyPeacefulToChakra(HeartChakra);
        ApplyPeacefulToChakra(ThroatChakra);
        ApplyPeacefulToChakra(ThirdEyeChakra);
        ApplyPeacefulToChakra(CrownChakra);
    }

    private void ApplyPeacefulToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.7f; // Slow, steady energy for peacefulness
            main.startSize = 0.3f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Moderate, steady emission for peace
        }
    }
}
