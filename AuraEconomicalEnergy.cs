using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraEconomicalEnergy : MonoBehaviour
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
        ApplyEconomicalEnergyToAura(EthericBody);
        ApplyEconomicalEnergyToAura(EmotionalBody);
        ApplyEconomicalEnergyToAura(MentalBody);
        ApplyEconomicalEnergyToAura(AstralBody);
        ApplyEconomicalEnergyToAura(EthericTemplate);
        ApplyEconomicalEnergyToAura(CelestialBody);
        ApplyEconomicalEnergyToAura(CausalBody);
    }

    private void ApplyEconomicalEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 0.7f; // Economical energy is used efficiently and slowly
            main.startSize = 0.5f; // Smaller particles due to conservation
            main.startLifetime = 9.0f; // Long-lasting as it's carefully used

            var emission = particleSystem.emission;
            emission.rateOverTime = 8f; // Economical energy is sparsely emitted
        }
    }
}
