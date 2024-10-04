using System.Collections.Generic;
using UnityEngine;

public class ElectronConfigurationGenerator : MonoBehaviour
{
    [Header("Electron Configuration")]
    public int atomicNumber;  // Set the atomic number of the element in Unity Inspector
    public List<ElectronSubshell> electronConfiguration = new List<ElectronSubshell>(); // List to store the configuration

    private int[] maxElectronsPerSubshell = { 2, 6, 10, 14 }; // s, p, d, f subshell electron capacities
    private string[] fillingOrder = { "1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p" };

    void Start()
    {
        GenerateElectronConfiguration(atomicNumber);
    }

    [ContextMenu("Generate Electron Configuration")]
    public void GenerateElectronConfiguration(int atomicNumber)
    {
        electronConfiguration.Clear();  // Clear previous configuration

        int remainingElectrons = atomicNumber;  // Total electrons to distribute

        foreach (string subshell in fillingOrder)
        {
            // Get the principal energy level and orbital type from the subshell (e.g., "1s" -> 1, 's')
            int energyLevel = int.Parse(subshell.Substring(0, 1));
            char orbitalType = subshell[1];
            int maxElectronsInSubshell = GetMaxElectronsInSubshell(orbitalType);

            // Determine how many electrons to place in this subshell
            int electronsInSubshell = Mathf.Min(remainingElectrons, maxElectronsInSubshell);

            // Add the subshell to the electron configuration
            electronConfiguration.Add(new ElectronSubshell(energyLevel, orbitalType, electronsInSubshell));

            // Subtract the electrons placed in this subshell from the remaining electrons
            remainingElectrons -= electronsInSubshell;

            // If no more electrons are left, break out of the loop
            if (remainingElectrons <= 0)
                break;
        }
    }

    // Helper function to get the maximum number of electrons in a subshell based on its orbital type (s, p, d, f)
    private int GetMaxElectronsInSubshell(char orbitalType)
    {
        switch (orbitalType)
        {
            case 's': return maxElectronsPerSubshell[0];  // s can hold 2 electrons
            case 'p': return maxElectronsPerSubshell[1];  // p can hold 6 electrons
            case 'd': return maxElectronsPerSubshell[2];  // d can hold 10 electrons
            case 'f': return maxElectronsPerSubshell[3];  // f can hold 14 electrons
            default: return 0;
        }
    }
}
