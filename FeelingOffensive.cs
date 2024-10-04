using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingOffensive : MonoBehaviour
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
        ApplyOffensiveFeeling();
    }

    public void ApplyOffensiveFeeling()
    {
        ApplyOffensiveToChakra(RootChakra);
        ApplyOffensiveToChakra(SacralChakra);
        ApplyOffensiveToChakra(SolarPlexusChakra);
        ApplyOffensiveToChakra(HeartChakra);
        ApplyOffensiveToChakra(ThroatChakra);
        ApplyOffensiveToChakra(ThirdEyeChakra);
        ApplyOffensiveToChakra(CrownChakra);
    }

    private void ApplyOffensiveToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.7f; // Fast, sharp energy for action and emotion
            main.startSize = 0.6f;
            main.startLifetime = 2.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // High energy bursts to reflect proactive, aggressive action
        }
    }
}
