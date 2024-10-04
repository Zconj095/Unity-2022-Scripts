using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FeelingRelief : MonoBehaviour
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
        ApplyReliefFeeling();
    }

    public void ApplyReliefFeeling()
    {
        ApplyReliefToChakra(RootChakra);
        ApplyReliefToChakra(SacralChakra);
        ApplyReliefToChakra(SolarPlexusChakra);
        ApplyReliefToChakra(HeartChakra);
        ApplyReliefToChakra(ThroatChakra);
        ApplyReliefToChakra(ThirdEyeChakra);
        ApplyReliefToChakra(CrownChakra);
    }

    private void ApplyReliefToChakra(GameObject chakra)
    {
        ParticleSystem particleSystem = chakra.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.9f; // Gentle, slow energy reflecting relief
            main.startSize = 0.4f;
            main.startLifetime = 3.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Gradual, steady energy flow to reflect the release of tension
        }
    }
}
