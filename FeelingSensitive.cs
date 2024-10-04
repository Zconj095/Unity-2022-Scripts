using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSensitive : MonoBehaviour
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
        ApplySensitiveFeeling();
    }

    public void ApplySensitiveFeeling()
    {
        ApplySensitiveToChakra(RootChakra);
        ApplySensitiveToChakra(SacralChakra);
        ApplySensitiveToChakra(SolarPlexusChakra);
        ApplySensitiveToChakra(HeartChakra);
        ApplySensitiveToChakra(ThroatChakra);
        ApplySensitiveToChakra(ThirdEyeChakra);
        ApplySensitiveToChakra(CrownChakra);
    }

    private void ApplySensitiveToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = Random.Range(1.0f, 1.8f); // Fluctuating energy to represent sensitivity
            main.startSize = Random.Range(0.4f, 0.7f); // Variable energy size for sensitivity
            main.startLifetime = 2.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = Random.Range(30f, 45f); // Fluctuating emission for intensity
        }
    }
}
