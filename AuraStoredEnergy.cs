using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraStoredEnergy : MonoBehaviour
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
        ApplyStoredEnergyToAura(EthericBody);
        ApplyStoredEnergyToAura(EmotionalBody);
        ApplyStoredEnergyToAura(MentalBody);
        ApplyStoredEnergyToAura(AstralBody);
        ApplyStoredEnergyToAura(EthericTemplate);
        ApplyStoredEnergyToAura(CelestialBody);
        ApplyStoredEnergyToAura(CausalBody);
    }

    private void ApplyStoredEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.5f; // Stored energy moves slower
            main.startSize = 0.7f;
            main.startLifetime = 8.0f; // Longer lifespan, conserved energy

            var emission = particleSystem.emission;
            emission.rateOverTime = 10f; // Slower, steadier release
        }
    }
}
