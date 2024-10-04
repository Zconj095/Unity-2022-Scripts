using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraNaturalEnergy : MonoBehaviour
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
        ApplyNaturalEnergyToAura(EthericBody);
        ApplyNaturalEnergyToAura(EmotionalBody);
        ApplyNaturalEnergyToAura(MentalBody);
        ApplyNaturalEnergyToAura(AstralBody);
        ApplyNaturalEnergyToAura(EthericTemplate);
        ApplyNaturalEnergyToAura(CelestialBody);
        ApplyNaturalEnergyToAura(CausalBody);
    }

    private void ApplyNaturalEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.2f; // Natural energy flows at a more dynamic pace
            main.startSize = 1.0f; // Larger particle size
            main.startLifetime = 5.0f; // Moderate lifespan

            var emission = particleSystem.emission;
            emission.rateOverTime = 30f; // Natural energy flows more frequently
        }
    }
}
