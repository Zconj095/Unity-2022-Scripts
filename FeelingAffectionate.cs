using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingAffectionate : MonoBehaviour
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
        ApplyAffectionateFeeling();
    }

    public void ApplyAffectionateFeeling()
    {
        ApplyAffectionateToChakra(RootChakra);
        ApplyAffectionateToChakra(SacralChakra);
        ApplyAffectionateToChakra(SolarPlexusChakra);
        ApplyAffectionateToChakra(HeartChakra);
        ApplyAffectionateToChakra(ThroatChakra);
        ApplyAffectionateToChakra(ThirdEyeChakra);
        ApplyAffectionateToChakra(CrownChakra);
    }

    private void ApplyAffectionateToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Soft, gentle energy for affection
            main.startSize = 0.3f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Slow, rhythmic emission for affectionate energy
        }
    }
}
