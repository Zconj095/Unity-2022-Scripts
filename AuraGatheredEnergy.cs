using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraGatheredEnergy : MonoBehaviour
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
        ApplyGatheredEnergyToAura(EthericBody);
        ApplyGatheredEnergyToAura(EmotionalBody);
        ApplyGatheredEnergyToAura(MentalBody);
        ApplyGatheredEnergyToAura(AstralBody);
        ApplyGatheredEnergyToAura(EthericTemplate);
        ApplyGatheredEnergyToAura(CelestialBody);
        ApplyGatheredEnergyToAura(CausalBody);
    }

    private void ApplyGatheredEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.8f; // Gathered energy flows slower
            main.startSize = 1.2f; // Large particles, showing it’s collected over time
            main.startLifetime = 7.0f; // Long-lasting due to accumulation

            var emission = particleSystem.emission;
            emission.rateOverTime = 12f; // Gradual release of gathered energy
        }
    }
}
