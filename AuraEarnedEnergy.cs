using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraEarnedEnergy : MonoBehaviour
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
        ApplyEarnedEnergyToAura(EthericBody);
        ApplyEarnedEnergyToAura(EmotionalBody);
        ApplyEarnedEnergyToAura(MentalBody);
        ApplyEarnedEnergyToAura(AstralBody);
        ApplyEarnedEnergyToAura(EthericTemplate);
        ApplyEarnedEnergyToAura(CelestialBody);
        ApplyEarnedEnergyToAura(CausalBody);
    }

    private void ApplyEarnedEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.0f; // Balanced, stable flow
            main.startSize = 0.9f; // Medium size
            main.startLifetime = 6.0f; // Lasts a moderate amount of time

            var emission = particleSystem.emission;
            emission.rateOverTime = 20f; // Earned energy flows at a steady pace
        }
    }
}
