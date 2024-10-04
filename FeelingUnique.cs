using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingUnique : MonoBehaviour
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
        ApplyUniqueFeeling();
    }

    public void ApplyUniqueFeeling()
    {
        ApplyUniqueToChakra(RootChakra);
        ApplyUniqueToChakra(SacralChakra);
        ApplyUniqueToChakra(SolarPlexusChakra);
        ApplyUniqueToChakra(HeartChakra);
        ApplyUniqueToChakra(ThroatChakra);
        ApplyUniqueToChakra(ThirdEyeChakra);
        ApplyUniqueToChakra(CrownChakra);
    }

    private void ApplyUniqueToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Non-linear, distinct energy for uniqueness
            main.startSize = Random.Range(0.4f, 0.7f); // Varied energy size to reflect uniqueness
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Abstract energy flow for a unique experience
        }
    }
}
