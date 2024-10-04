using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingCalm : MonoBehaviour
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
        ApplyCalmFeeling();
    }

    public void ApplyCalmFeeling()
    {
        ApplyCalmToChakra(RootChakra);
        ApplyCalmToChakra(SacralChakra);
        ApplyCalmToChakra(SolarPlexusChakra);
        ApplyCalmToChakra(HeartChakra);
        ApplyCalmToChakra(ThroatChakra);
        ApplyCalmToChakra(ThirdEyeChakra);
        ApplyCalmToChakra(CrownChakra);
    }

    private void ApplyCalmToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.7f; // Slow and smooth energy for calmness
            main.startSize = 0.4f;
            main.startLifetime = 4.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Steady emission for calm energy
        }
    }
}
