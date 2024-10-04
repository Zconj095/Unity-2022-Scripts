using UnityEngine;

public enum EnergySubType
{
    Borrowed,
    Gathered,
    Earned,
    Obtained,
    Economical,
    Stored,
}

public enum EnergyMode
{
    Willed,
    Forced
}

public enum EnergyType
{
    Natural,
    Artificial
}

[System.Serializable]
public class EnergySystem
{
    [SerializeField] private EnergyType energyType;
    [SerializeField] private EnergyMode energyMode;
    [SerializeField] private EnergySubType subType;
    [SerializeField] private float amount;
    [SerializeField] private float maxAmount;

    public EnergyType Type => energyType;
    public EnergyMode Mode => energyMode;
    public EnergySubType SubType => subType;
    public float Amount => amount;
    public float MaxAmount => maxAmount;

    public EnergySystem(EnergyType type, EnergyMode mode, EnergySubType subType, float initialAmount, float maxAmount)
    {
        this.energyType = type;
        this.energyMode = mode;
        this.subType = subType;
        this.amount = initialAmount;
        this.maxAmount = maxAmount;
    }

    public void UseEnergy(float usageAmount)
    {
        if (amount >= usageAmount)
        {
            amount -= usageAmount;
            Debug.Log($"{usageAmount} of {subType} energy used. Remaining: {amount}");
        }
        else
        {
            Debug.LogWarning("Not enough energy to use.");
        }
    }

    public void RechargeEnergy(float rechargeAmount)
    {
        if (amount + rechargeAmount <= maxAmount)
        {
            amount += rechargeAmount;
            Debug.Log($"{rechargeAmount} of {subType} energy recharged. Current: {amount}");
        }
        else
        {
            amount = maxAmount;
            Debug.Log($"Energy fully recharged to {maxAmount}");
        }
    }
}
