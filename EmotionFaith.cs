using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionFaith : MonoBehaviour
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
        ApplyFaithEmotion();
    }

    public void ApplyFaithEmotion()
    {
        ApplyFaithToChakra(RootChakra);
        ApplyFaithToChakra(SacralChakra);
        ApplyFaithToChakra(SolarPlexusChakra);
        ApplyFaithToChakra(HeartChakra);
        ApplyFaithToChakra(ThroatChakra);
        ApplyFaithToChakra(ThirdEyeChakra);
        ApplyFaithToChakra(CrownChakra);
    }

    private void ApplyFaithToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Steady, continuous energy reflecting faith
            main.startSize = 0.5f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Strong, continuous flow symbolizing commitment and belief
        }
    }
}
