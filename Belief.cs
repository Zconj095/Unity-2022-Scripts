using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public class Belief
{
    [SerializeField] private string description;
    [SerializeField] private bool isActive;

    public string Description => description;
    public bool IsActive => isActive;

    public Belief(string description)
    {
        this.description = description;
        isActive = false;
    }

    public void SafeguardBelief()
    {
        isActive = true;
        Debug.Log($"Belief safeguarded: {description}");
    }

    public void AbsorbBelief(Belief otherBelief)
    {
        Debug.Log($"Absorbed belief: {otherBelief.Description}");
    }

    public void ReverseNegativeEnergy(EnergySystem negativeEnergy)
    {
        if (negativeEnergy.Amount > 0)
        {
            negativeEnergy.UseEnergy(negativeEnergy.Amount);
            Debug.Log($"Reversed {negativeEnergy.Amount} of negative energy.");
        }
        else
        {
            Debug.LogWarning("No negative energy to reverse.");
        }
    }

    public void StoreBeliefInObject(GameObject obj)
    {
        Debug.Log($"Stored belief '{description}' in {obj.name}");
        obj.AddComponent<BeliefStorage>().StoreBelief(this);
    }
}

public class BeliefStorage : MonoBehaviour
{
    [SerializeField] private Belief storedBelief;

    public void StoreBelief(Belief belief)
    {
        storedBelief = belief;
        Debug.Log($"Belief '{belief.Description}' stored in object {gameObject.name}");
    }
}
