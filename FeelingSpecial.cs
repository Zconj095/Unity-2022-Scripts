using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingSpecial : MonoBehaviour
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
        ApplySpecialFeeling();
    }

    public void ApplySpecialFeeling()
    {
        ApplySpecialToChakra(RootChakra);
        ApplySpecialToChakra(SacralChakra);
        ApplySpecialToChakra(SolarPlexusChakra);
        ApplySpecialToChakra(HeartChakra);
        ApplySpecialToChakra(ThroatChakra);
        ApplySpecialToChakra(ThirdEyeChakra);
        ApplySpecialToChakra(CrownChakra);
    }

    private void ApplySpecialToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Unique and non-linear energy for individuality
            main.startSize = Random.Range(0.3f, 0.6f); // Abstract flow for uniqueness
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Non-uniform energy flow to reflect specialness
        }
    }
}
