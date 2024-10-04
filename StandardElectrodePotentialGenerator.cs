using System.Collections.Generic;
using UnityEngine;

public class StandardElectrodePotentialGenerator : MonoBehaviour
{
    [Header("Standard Electrode Potentials")]
    public List<ElectrodePotential> electrodePotentials = new List<ElectrodePotential>();  // List to store electrode potentials

    void Start()
    {
        GenerateStandardElectrodePotentials();
    }

    [ContextMenu("Generate Standard Electrode Potentials")]
    public void GenerateStandardElectrodePotentials()
    {
        electrodePotentials.Clear();  // Clear the list before generating

        // Add common reduction half-reactions and their standard electrode potentials (in volts)
        electrodePotentials.Add(new ElectrodePotential("Li⁺ + e⁻ → Li", -3.04f));  // Most negative potential
        electrodePotentials.Add(new ElectrodePotential("K⁺ + e⁻ → K", -2.93f));
        electrodePotentials.Add(new ElectrodePotential("Ca²⁺ + 2e⁻ → Ca", -2.87f));
        electrodePotentials.Add(new ElectrodePotential("Na⁺ + e⁻ → Na", -2.71f));
        electrodePotentials.Add(new ElectrodePotential("Mg²⁺ + 2e⁻ → Mg", -2.37f));
        electrodePotentials.Add(new ElectrodePotential("Be²⁺ + 2e⁻ → Be", -1.85f));
        electrodePotentials.Add(new ElectrodePotential("Al³⁺ + 3e⁻ → Al", -1.66f));
        electrodePotentials.Add(new ElectrodePotential("Mn²⁺ + 2e⁻ → Mn", -1.18f));
        electrodePotentials.Add(new ElectrodePotential("Zn²⁺ + 2e⁻ → Zn", -0.76f));
        electrodePotentials.Add(new ElectrodePotential("Cr³⁺ + 3e⁻ → Cr", -0.74f));
        electrodePotentials.Add(new ElectrodePotential("Fe²⁺ + 2e⁻ → Fe", -0.44f));
        electrodePotentials.Add(new ElectrodePotential("Cd²⁺ + 2e⁻ → Cd", -0.40f));
        electrodePotentials.Add(new ElectrodePotential("Ni²⁺ + 2e⁻ → Ni", -0.25f));
        electrodePotentials.Add(new ElectrodePotential("Sn²⁺ + 2e⁻ → Sn", -0.14f));
        electrodePotentials.Add(new ElectrodePotential("Pb²⁺ + 2e⁻ → Pb", -0.13f));
        electrodePotentials.Add(new ElectrodePotential("H⁺ + e⁻ → ½ H₂", 0.00f));  // Standard hydrogen electrode
        electrodePotentials.Add(new ElectrodePotential("Sb³⁺ + 3e⁻ → Sb", 0.15f));
        electrodePotentials.Add(new ElectrodePotential("Bi³⁺ + 3e⁻ → Bi", 0.20f));
        electrodePotentials.Add(new ElectrodePotential("Cu²⁺ + 2e⁻ → Cu", 0.34f));
        electrodePotentials.Add(new ElectrodePotential("I₂ + 2e⁻ → 2I⁻", 0.54f));
        electrodePotentials.Add(new ElectrodePotential("Ag⁺ + e⁻ → Ag", 0.80f));
        electrodePotentials.Add(new ElectrodePotential("Hg²⁺ + 2e⁻ → Hg", 0.85f));
        electrodePotentials.Add(new ElectrodePotential("Pt²⁺ + 2e⁻ → Pt", 1.20f));
        electrodePotentials.Add(new ElectrodePotential("Au³⁺ + 3e⁻ → Au", 1.50f));
        electrodePotentials.Add(new ElectrodePotential("F₂ + 2e⁻ → 2F⁻", 2.87f));  // Highly reactive nonmetal (oxidizing agent)
        electrodePotentials.Add(new ElectrodePotential("Cl₂ + 2e⁻ → 2Cl⁻", 1.36f));
        electrodePotentials.Add(new ElectrodePotential("Br₂ + 2e⁻ → 2Br⁻", 1.07f));
        electrodePotentials.Add(new ElectrodePotential("I₂ + 2e⁻ → 2I⁻", 0.54f));
        
        // Additional electrode potentials for completeness
        electrodePotentials.Add(new ElectrodePotential("O₂ + 4H⁺ + 4e⁻ → 2H₂O", 1.23f));  // Oxygen reduction
        electrodePotentials.Add(new ElectrodePotential("MnO₄⁻ + 8H⁺ + 5e⁻ → Mn²⁺ + 4H₂O", 1.51f));  // Permanganate reduction
        electrodePotentials.Add(new ElectrodePotential("NO₃⁻ + 4H⁺ + 3e⁻ → NO + 2H₂O", 0.96f));  // Nitric acid reduction
        electrodePotentials.Add(new ElectrodePotential("Cr₂O₇²⁻ + 14H⁺ + 6e⁻ → 2Cr³⁺ + 7H₂O", 1.33f));  // Dichromate reduction
        electrodePotentials.Add(new ElectrodePotential("SO₄²⁻ + 4H⁺ + 2e⁻ → SO₂ + 2H₂O", 0.20f));  // Sulfate reduction

    }
}
