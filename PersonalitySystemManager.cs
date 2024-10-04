using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum PersonalityTypes
{
    ISFJ, ISFP, ISTP, ISTJ,
    INFP, INFJ, INTP, INTJ,
    ESFJ, ESFP, ESTP, ESTJ,
    ENFP, ENFJ, ENTP, ENTJ
}

public enum CognitiveFunctions
{
    IntrovertedIntuition,
    IntrovertedFeeling,
    IntrovertedSensing,
    IntrovertedThinking,
    ExtrovertedIntuition,
    ExtrovertedFeeling,
    ExtrovertedSensing,
    ExtrovertedThinking
}

public enum FunctionType
{
    Dominant,
    Primary,
    Secondary,
    Auxiliary,
    Tertiary,
    Passive
}


public class PersonalitySystemManager : MonoBehaviour
{
    public PersonalityType personalityType;
    public CognitiveFunctions cognitiveFunction;
    public FunctionType functionType;

    public GameObject chakraPrefab; // Prefab for chakras
    public GameObject auraPrefab;   // Prefab for auras

    // Variables to store the initial values of speed, size, and lifetime
    private float currentSpeed;
    private float currentSize;
    private float currentLifetime;

    void Start()
    {
        AttachChakraPrefab(chakraPrefab);
        AttachAuraPrefab(auraPrefab);

        ApplyPersonalityEffects();
        ApplyCognitiveFunctionEffects();
        ApplyFunctionTypeEffects();  // Apply the modifier based on Function Type
    }

    private void AttachChakraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Chakra Prefab attached for Personality: " + personalityType + ", Cognitive Function: " + cognitiveFunction);
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
            Debug.Log("Aura Prefab attached for Personality: " + personalityType + ", Cognitive Function: " + cognitiveFunction);
        }
        else
        {
            Debug.LogError("Aura Prefab is missing!");
        }
    }

    private void ApplyPersonalityEffects()
    {
        // Adjust chakra/aura particle system based on the personality type
        switch (personalityType)
        {
            case PersonalityType.ISFJ:
                currentSpeed = 1.0f;
                currentSize = 1.2f;
                currentLifetime = 5.0f;
                break;
            case PersonalityType.ENTP:
                currentSpeed = 1.5f;
                currentSize = 1.0f;
                currentLifetime = 4.0f;
                break;
            case PersonalityType.INFJ:
                currentSpeed = 1.3f;
                currentSize = 1.1f;
                currentLifetime = 6.0f;
                break;
            case PersonalityType.ISTP:
                currentSpeed = 1.1f;
                currentSize = 1.2f;
                currentLifetime = 5.5f;
                break;
            case PersonalityType.INTJ:
                currentSpeed = 1.4f;
                currentSize = 1.1f;
                currentLifetime = 5.0f;
                break;
            case PersonalityType.ENFP:
                currentSpeed = 1.6f;
                currentSize = 1.0f;
                currentLifetime = 4.0f;
                break;
            case PersonalityType.ISFP:
                currentSpeed = 1.0f;
                currentSize = 1.2f;
                currentLifetime = 5.0f;
                break;
            case PersonalityType.ESFJ:
                currentSpeed = 1.3f;
                currentSize = 1.0f;
                currentLifetime = 5.5f;
                break;
            case PersonalityType.ENTJ:
                currentSpeed = 1.7f;
                currentSize = 1.1f;
                currentLifetime = 4.5f;
                break;
            case PersonalityType.ESFP:
                currentSpeed = 1.5f;
                currentSize = 1.0f;
                currentLifetime = 4.0f;
                break;
            case PersonalityType.ESTJ:
                currentSpeed = 1.4f;
                currentSize = 1.1f;
                currentLifetime = 5.0f;
                break;
            case PersonalityType.ENFJ:
                currentSpeed = 1.7f;
                currentSize = 1.0f;
                currentLifetime = 4.5f;
                break;
            case PersonalityType.INFP:
                currentSpeed = 1.1f;
                currentSize = 1.2f;
                currentLifetime = 6.5f;
                break;
            case PersonalityType.ESTP:
                currentSpeed = 1.5f;
                currentSize = 1.0f;
                currentLifetime = 4.0f;
                break;
            case PersonalityType.INTP:
                currentSpeed = 1.3f;
                currentSize = 1.1f;
                currentLifetime = 5.0f;
                break;
            case PersonalityType.ISTJ:  // This is the missing type
                currentSpeed = 1.2f;
                currentSize = 1.2f;
                currentLifetime = 5.5f;
                break;
        }
        AdjustChakraAuraValues(currentSpeed, currentSize, currentLifetime);
    }

    private void ApplyCognitiveFunctionEffects()
    {
        // Modify chakra/aura behavior based on cognitive function
        switch (cognitiveFunction)
        {
            case CognitiveFunctions.IntrovertedIntuition:
                currentSpeed = 1.2f;
                currentSize = 1.1f;
                currentLifetime = 6.5f;
                break;
            case CognitiveFunctions.IntrovertedFeeling:
                currentSpeed = 1.0f;
                currentSize = 1.2f;
                currentLifetime = 6.0f;
                break;
            case CognitiveFunctions.IntrovertedSensing:
                currentSpeed = 1.1f;
                currentSize = 1.2f;
                currentLifetime = 5.5f;
                break;
            case CognitiveFunctions.IntrovertedThinking:
                currentSpeed = 1.3f;
                currentSize = 1.1f;
                currentLifetime = 5.0f;
                break;
            case CognitiveFunctions.ExtrovertedIntuition:
                currentSpeed = 1.5f;
                currentSize = 1.0f;
                currentLifetime = 4.0f;
                break;
            case CognitiveFunctions.ExtrovertedFeeling:
                currentSpeed = 1.4f;
                currentSize = 1.0f;
                currentLifetime = 4.5f;
                break;
            case CognitiveFunctions.ExtrovertedSensing:
                currentSpeed = 1.6f;
                currentSize = 1.0f;
                currentLifetime = 4.0f;
                break;
            case CognitiveFunctions.ExtrovertedThinking:
                currentSpeed = 1.5f;
                currentSize = 1.0f;
                currentLifetime = 5.0f;
                break;
        }
        AdjustChakraAuraValues(currentSpeed, currentSize, currentLifetime);
    }

    private void ApplyFunctionTypeEffects()
    {
        float modifier = 1.0f; // Default modifier

        // Apply different modifiers based on function type
        switch (functionType)
        {
            case FunctionType.Dominant:
                modifier = 1.2f; // Increase effect for Dominant function
                break;
            case FunctionType.Auxiliary:
                modifier = 1.1f; // Slight increase for Auxiliary function
                break;
            case FunctionType.Tertiary:
                modifier = 0.9f; // Slight decrease for Tertiary function
                break;
            case FunctionType.Passive:
                modifier = 0.8f; // Decrease for Passive function
                break;
        }

        // Apply the modifier to the current chakra and aura particle system values
        AdjustChakraAuraValues(currentSpeed * modifier, currentSize * modifier, currentLifetime * modifier);
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

