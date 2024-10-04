using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmotionJoy : MonoBehaviour
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
        ApplyJoyEmotion();
    }

    public void ApplyJoyEmotion()
    {
        ApplyJoyToChakra(RootChakra);
        ApplyJoyToChakra(SacralChakra);
        ApplyJoyToChakra(SolarPlexusChakra);
        ApplyJoyToChakra(HeartChakra);
        ApplyJoyToChakra(ThroatChakra);
        ApplyJoyToChakra(ThirdEyeChakra);
        ApplyJoyToChakra(CrownChakra);
    }

    private void ApplyJoyToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.7f; // Fast, bright energy for overwhelming joy
            main.startSize = 0.6f;
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // High energy output to reflect immense happiness and satisfaction
        }
    }
}
