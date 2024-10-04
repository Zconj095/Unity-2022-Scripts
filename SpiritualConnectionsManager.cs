using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum SpiritualConnection
{
    Mind,
    Body,
    Soul,
    Spirit
}

public enum EnergyState
{
    Mental,
    Emotional,
    Physical,
    Spiritual
}

public enum EnergyTypes
{
    Yin,
    Yang,
    Ming,
    Ying
}



public class SpiritualConnectionsManager : MonoBehaviour
{
    public SpiritualConnection spiritualConnection;
    public EnergyState energyState;
    public EnergyTypes kiEnergyType;
    public EnergyTypes chiEnergyType;

    public GameObject chakraPrefab; // Prefab for chakras
    public GameObject auraPrefab;   // Prefab for auras

    void Start()
    {
        AttachChakraPrefab(chakraPrefab);
        AttachAuraPrefab(auraPrefab);

        ApplySpiritualConnectionEffects();
        ApplyEnergyStateEffects();
        ApplyKiChiEffects();
    }

    private void AttachChakraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Chakra Prefab attached for Spiritual Connection: " + spiritualConnection + ", Energy State: " + energyState);
        }
        else
        {
            Debug.LogError("Chakra Prefab is missing!");
        }
    }

    private void AttachAuraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Aura Prefab attached for Spiritual Connection: " + spiritualConnection + ", Energy State: " + energyState);
        }
        else
        {
            Debug.LogError("Aura Prefab is missing!");
        }
    }

    private void ApplySpiritualConnectionEffects()
    {
        // Example of how each spiritual connection affects the system
        switch (spiritualConnection)
        {
            case SpiritualConnection.Mind:
                AdjustChakraAuraValues(1.1f, 1.0f, 5.0f); // Faster energy flow
                break;
            case SpiritualConnection.Body:
                AdjustChakraAuraValues(0.9f, 1.3f, 6.5f); // Grounded energy
                break;
            case SpiritualConnection.Soul:
                AdjustChakraAuraValues(1.2f, 1.1f, 4.0f); // Fluid, soul-connected flow
                break;
            case SpiritualConnection.Spirit:
                AdjustChakraAuraValues(1.5f, 1.0f, 6.0f); // Higher spiritual vibration
                break;
        }
    }

    private void ApplyEnergyStateEffects()
    {
        // Example of how each energy state affects the system
        switch (energyState)
        {
            case EnergyState.Mental:
                AdjustChakraAuraValues(1.0f, 1.1f, 4.5f); // Mental clarity effect
                break;
            case EnergyState.Emotional:
                AdjustChakraAuraValues(0.8f, 1.0f, 6.0f); // Emotional depth effect
                break;
            case EnergyState.Physical:
                AdjustChakraAuraValues(1.2f, 1.3f, 5.0f); // Physical energy flow
                break;
            case EnergyState.Spiritual:
                AdjustChakraAuraValues(1.3f, 1.0f, 6.5f); // Spiritual connection effect
                break;
        }
    }

    private void ApplyKiChiEffects()
    {
        // Adjust chakra and aura based on Ki energy (Yin, Yang, etc.)
        ApplyEnergyTypeEffect(kiEnergyType, "Ki");
        ApplyEnergyTypeEffect(chiEnergyType, "Chi");
    }

    private void ApplyEnergyTypeEffect(EnergyTypes EnergyTypes, string energySource)
    {
        switch (EnergyTypes)
        {
            case EnergyTypes.Yin:
                Debug.Log(energySource + " Energy: Yin - Slow and deep energy");
                AdjustChakraAuraValues(0.8f, 1.2f, 6.0f);
                break;
            case EnergyTypes.Yang:
                Debug.Log(energySource + " Energy: Yang - Fast and vibrant energy");
                AdjustChakraAuraValues(1.5f, 1.0f, 4.5f);
                break;
            case EnergyTypes.Ming:
                Debug.Log(energySource + " Energy: Ming - Balanced, harmonized energy");
                AdjustChakraAuraValues(1.2f, 1.1f, 5.0f);
                break;
            case EnergyTypes.Ying:
                Debug.Log(energySource + " Energy: Ying - Dynamic and transformative energy");
                AdjustChakraAuraValues(1.3f, 1.0f, 5.5f);
                break;
        }
    }

    private void AdjustChakraAuraValues(float speed, float size, float lifetime)
    {
        // Adjust values for the chakra and aura particle systems
        if (chakraPrefab != null)
        {
            ParticleSystem[] chakraParticleSystems = chakraPrefab.GetComponentsInChildren<ParticleSystem>();
            foreach (ParticleSystem ps in chakraParticleSystems)
            {
                var mainModule = ps.main;
                mainModule.simulationSpeed = speed;
                mainModule.startSize = size;
                mainModule.startLifetime = lifetime;
            }
        }

        if (auraPrefab != null)
        {
            ParticleSystem[] auraParticleSystems = auraPrefab.GetComponentsInChildren<ParticleSystem>();
            foreach (ParticleSystem ps in auraParticleSystems)
            {
                var mainModule = ps.main;
                mainModule.simulationSpeed = speed;
                mainModule.startSize = size;
                mainModule.startLifetime = lifetime;
            }
        }
    }
}
