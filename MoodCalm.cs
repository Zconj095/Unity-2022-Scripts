using UnityEngine;

public class MoodCalm : MonoBehaviour
{
    // Attach existing chakra GameObjects here in the Unity inspector
    public GameObject RootChakra;
    public GameObject SacralChakra;
    public GameObject SolarPlexusChakra;
    public GameObject HeartChakra;
    public GameObject ThroatChakra;
    public GameObject ThirdEyeChakra;
    public GameObject CrownChakra;

    void Start()
    {
        ApplyCalmMood();
    }

    public void ApplyCalmMood()
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
            main.simulationSpeed = 0.5f; // Calm effect: slow down particles
            main.startSize = 0.2f;
            main.startLifetime = 3f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 10f; // Calm emission rate for all chakras
        }
    }
}
