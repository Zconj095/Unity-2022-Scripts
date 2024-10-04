using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AuraArtificialEnergy : MonoBehaviour
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
        ApplyArtificialEnergyToAura(EthericBody);
        ApplyArtificialEnergyToAura(EmotionalBody);
        ApplyArtificialEnergyToAura(MentalBody);
        ApplyArtificialEnergyToAura(AstralBody);
        ApplyArtificialEnergyToAura(EthericTemplate);
        ApplyArtificialEnergyToAura(CelestialBody);
        ApplyArtificialEnergyToAura(CausalBody);
    }

    private void ApplyArtificialEnergyToAura(GameObject auraLayer)
    {
        ParticleSystem particleSystem = auraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;
            main.simulationSpeed = 1.5f; // Artificial energy is synthetic and faster
            main.startSize = 0.6f; // Smaller particle size for artificial nature
            main.startLifetime = 4.0f; // Shorter lifespan

            var emission = particleSystem.emission;
            emission.rateOverTime = 40f; // Higher frequency due to artificial stimulation
        }
    }
}
