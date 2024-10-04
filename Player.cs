using UnityEngine;

public class Player : MonoBehaviour
{
    [Header("Energy Settings")]
    [SerializeField] private EnergySystem currentEnergy;

    void Start()
    {
        // Set up a default energy system (Natural, Willed, Gathered)
        currentEnergy = new EnergySystem(EnergyType.Natural, EnergyMode.Willed, EnergySubType.Gathered, 100f, 200f);

        // Test energy usage
        currentEnergy.UseEnergy(30f);
        currentEnergy.RechargeEnergy(50f);
    }

    public void SwitchEnergy(EnergyType energyType, EnergyMode energyMode, EnergySubType subType, float initialAmount, float maxAmount)
    {
        // Ensure only one energy type and mode are active
        currentEnergy = new EnergySystem(energyType, energyMode, subType, initialAmount, maxAmount);
        Debug.Log($"Switched energy to: {energyType} with mode: {energyMode}");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.U))
        {
            currentEnergy.UseEnergy(10f);
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            currentEnergy.RechargeEnergy(20f);
        }

        if (Input.GetKeyDown(KeyCode.N)) // Switch to Natural, Willed, Earned energy
        {
            SwitchEnergy(EnergyType.Natural, EnergyMode.Willed, EnergySubType.Earned, 80f, 150f);
        }

        if (Input.GetKeyDown(KeyCode.A)) // Switch to Artificial, Forced, Stored energy
        {
            SwitchEnergy(EnergyType.Artificial, EnergyMode.Forced, EnergySubType.Stored, 50f, 100f);
        }
    }
}
