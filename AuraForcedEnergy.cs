using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class AuraForcedEnergy : MonoBehaviour
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
        ApplyForcedEnergyToAura(EthericBody);
        ApplyForcedEnergyToAura(EmotionalBody);
        ApplyForcedEnergyToAura(MentalBody);
        ApplyForcedEnergyToAura(AstralBody);
        ApplyForcedEnergyToAura(EthericTemplate);
        ApplyForcedEnergyToAura(CelestialBody);
        ApplyForcedEnergyToAura(CausalBody);
    }

    private void ApplyForcedEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 2.0f; // Forced energy is faster and more erratic
            main.startSize = 1.2f;
            main.startLifetime = 4.0f; // Shorter lifespan due to instability

            var emission = particleSystem.emission;
            emission.rateOverTime = 50f; // Forced energy is abundant but chaotic
        }
    }
}
