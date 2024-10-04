using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoodTalkative : MonoBehaviour
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
        ApplyTalkativeMood();
    }

    public void ApplyTalkativeMood()
    {
        ApplyTalkativeToChakra(RootChakra);
        ApplyTalkativeToChakra(SacralChakra);
        ApplyTalkativeToChakra(SolarPlexusChakra);
        ApplyTalkativeToChakra(HeartChakra);
        ApplyTalkativeToChakra(ThroatChakra);
        ApplyTalkativeToChakra(ThirdEyeChakra);
        ApplyTalkativeToChakra(CrownChakra);
    }

    private void ApplyTalkativeToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Fast, active energy for talkative behavior
            main.startSize = 0.5f;
            main.startLifetime = 2.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // High emission rate for energetic, talkative behavior
        }
    }
}
