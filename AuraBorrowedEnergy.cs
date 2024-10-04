using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraBorrowedEnergy : MonoBehaviour
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
        ApplyBorrowedEnergyToAura(EthericBody);
        ApplyBorrowedEnergyToAura(EmotionalBody);
        ApplyBorrowedEnergyToAura(MentalBody);
        ApplyBorrowedEnergyToAura(AstralBody);
        ApplyBorrowedEnergyToAura(EthericTemplate);
        ApplyBorrowedEnergyToAura(CelestialBody);
        ApplyBorrowedEnergyToAura(CausalBody);
    }

    private void ApplyBorrowedEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.3f; // Borrowed energy moves quickly
            main.startSize = 0.8f;
            main.startLifetime = 5.0f; // Short lifespan as it needs to be paid back

            var emission = particleSystem.emission;
            emission.rateOverTime = 35f; // Fast emission, borrowed energy is temporary
        }
    }
}
