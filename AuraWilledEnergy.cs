using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraWilledEnergy : MonoBehaviour
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
        ApplyWilledEnergyToAura(EthericBody);
        ApplyWilledEnergyToAura(EmotionalBody);
        ApplyWilledEnergyToAura(MentalBody);
        ApplyWilledEnergyToAura(AstralBody);
        ApplyWilledEnergyToAura(EthericTemplate);
        ApplyWilledEnergyToAura(CelestialBody);
        ApplyWilledEnergyToAura(CausalBody);
    }

    private void ApplyWilledEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Controlled flow for Willed energy
            main.startSize = 0.8f;
            main.startLifetime = 6.0f;

            var emission = particleSystem.emission;
            emission.rateOverTime = 15f; // Willed energy is more controlled and steady
        }
    }
}
