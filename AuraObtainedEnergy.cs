using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraObtainedEnergy : MonoBehaviour
{
    public GameObject EthericBody;
    public GameObject EmotionalBody;
    public GameObject MentalBody;
    public GameObject AstralBody;
    public GameObject EthericTemplate;
    public GameObject CelestialBody;
    public GameObject CausalBody;

    void Start()
    {
        ApplyContentAura();
    }

    public void ApplyContentAura()
    {
        ApplyObtainedEnergyToAura(EthericBody);
        ApplyObtainedEnergyToAura(EmotionalBody);
        ApplyObtainedEnergyToAura(MentalBody);
        ApplyObtainedEnergyToAura(AstralBody);
        ApplyObtainedEnergyToAura(EthericTemplate);
        ApplyObtainedEnergyToAura(CelestialBody);
        ApplyObtainedEnergyToAura(CausalBody);
    }

    private void ApplyObtainedEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Obtained energy moves faster than natural
            main.startSize = 0.9f;
            main.startLifetime = 4.5f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 25f; // Moderate flow of obtained energy
        }
    }
}
