using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public enum BeliefStructure
{
    Weld,
    Variable,
    Hierarchical,
    Synapse
}

public enum InnerOuter
{
    Inner,
    Outer
}

public class BeliefSystemManager : MonoBehaviour
{
    public BeliefStructure beliefStructure;
    public InnerOuter belief;  // Belief type (Inner or Outer)
    public InnerOuter faith;   // Faith type (Inner or Outer)
    public InnerOuter virtue;  // Virtue type (Inner or Outer)
    public InnerOuter moral;   // Moral type (Inner or Outer)
    public InnerOuter goal;    // Goal type (Inner or Outer)

    public GameObject chakraPrefab;  // Prefab for chakras
    public GameObject auraPrefab;    // Prefab for auras

    void Start()
    {
        AttachChakraPrefab(chakraPrefab);
        AttachAuraPrefab(auraPrefab);

        ApplyBeliefStructureEffects();
        ApplyBeliefEffects();
        ApplyFaithEffects();
        ApplyVirtueEffects();
        ApplyMoralEffects();
        ApplyGoalEffects();
    }

    private void AttachChakraPrefab(GameObject prefab)
    {
        if (prefab != null)
        {
            Debug.Log("Chakra Prefab attached for Belief Structure: " + beliefStructure);
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
            Debug.Log("Aura Prefab attached for Belief Structure: " + beliefStructure);
        }
        else
        {
            Debug.LogError("Aura Prefab is missing!");
        }
    }

    private void ApplyBeliefStructureEffects()
    {
        // Example: Each belief structure affects the chakra/aura behavior
        switch (beliefStructure)
        {
            case BeliefStructure.Weld:
                AdjustChakraAuraValues(1.2f, 1.1f, 5.0f); // Welded beliefs are stable
                break;
            case BeliefStructure.Variable:
                AdjustChakraAuraValues(1.0f, 1.0f, 6.0f); // Variable beliefs are more adaptable
                break;
            case BeliefStructure.Hierarchical:
                AdjustChakraAuraValues(1.3f, 1.1f, 4.5f); // Hierarchical beliefs provide structured energy
                break;
            case BeliefStructure.Synapse:
                AdjustChakraAuraValues(1.5f, 1.2f, 4.0f); // Synapse beliefs are dynamic and fast
                break;
        }
    }

    private void ApplyBeliefEffects()
    {
        // Example: Inner or Outer beliefs affect chakra/aura behavior
        switch (belief)
        {
            case InnerOuter.Inner:
                AdjustChakraAuraValues(1.1f, 1.2f, 5.5f); // Inner beliefs affect core energies
                break;
            case InnerOuter.Outer:
                AdjustChakraAuraValues(1.2f, 1.0f, 5.0f); // Outer beliefs affect external energies
                break;
        }
    }

    private void ApplyFaithEffects()
    {
        // Example: Inner or Outer faiths modify behavior
        switch (faith)
        {
            case InnerOuter.Inner:
                AdjustChakraAuraValues(1.0f, 1.2f, 6.0f); // Inner faith affects depth of energy
                break;
            case InnerOuter.Outer:
                AdjustChakraAuraValues(1.3f, 1.0f, 5.0f); // Outer faith affects outward flow of energy
                break;
        }
    }

    private void ApplyVirtueEffects()
    {
        // Example: Inner or Outer virtues
        switch (virtue)
        {
            case InnerOuter.Inner:
                AdjustChakraAuraValues(1.1f, 1.1f, 6.5f); // Inner virtue affects disciplined energy
                break;
            case InnerOuter.Outer:
                AdjustChakraAuraValues(1.2f, 1.0f, 5.5f); // Outer virtue affects external, action-based energy
                break;
        }
    }

    private void ApplyMoralEffects()
    {
        // Example: Inner or Outer morals
        switch (moral)
        {
            case InnerOuter.Inner:
                AdjustChakraAuraValues(1.0f, 1.2f, 6.5f); // Inner morals guide internal energy balance
                break;
            case InnerOuter.Outer:
                AdjustChakraAuraValues(1.3f, 1.0f, 5.0f); // Outer morals guide interaction with the outside world
                break;
        }
    }

    private void ApplyGoalEffects()
    {
        // Example: Inner or Outer goals
        switch (goal)
        {
            case InnerOuter.Inner:
                AdjustChakraAuraValues(1.2f, 1.1f, 6.0f); // Inner goals influence long-term focus of energy
                break;
            case InnerOuter.Outer:
                AdjustChakraAuraValues(1.3f, 1.0f, 5.0f); // Outer goals affect short-term, action-based energy
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
