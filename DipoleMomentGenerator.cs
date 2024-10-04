using System.Collections.Generic;
using UnityEngine;

public class DipoleMomentGenerator : MonoBehaviour
{
    [Header("Dipole Moments of Molecules")]
    public List<DipoleMomentMolecule> dipoleMomentMolecules = new List<DipoleMomentMolecule>();

    void Start()
    {
        GenerateDipoleMoments();
    }

    [ContextMenu("Generate Dipole Moments")]
    public void GenerateDipoleMoments()
    {
        dipoleMomentMolecules.Clear();  // Clear the list before generating

        // Add common molecules and their dipole moments (in Debye units, D)
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Water", "H₂O", 1.85f));      // Water
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Hydrogen Chloride", "HCl", 1.08f));  // HCl
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Ammonia", "NH₃", 1.47f));    // Ammonia
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Carbon Dioxide", "CO₂", 0.0f)); // CO₂ (non-polar)
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Methanol", "CH₃OH", 1.69f)); // Methanol
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Ethanol", "C₂H₅OH", 1.69f)); // Ethanol
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Acetone", "C₃H₆O", 2.88f));  // Acetone
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Hydrogen Sulfide", "H₂S", 0.97f)); // H₂S
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Carbon Tetrachloride", "CCl₄", 0.0f)); // CCl₄ (non-polar)
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Chloroform", "CHCl₃", 1.01f));  // Chloroform
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Formaldehyde", "CH₂O", 2.33f)); // Formaldehyde
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Hydrogen Fluoride", "HF", 1.82f)); // Hydrogen Fluoride
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Benzene", "C₆H₆", 0.0f));     // Benzene (non-polar)
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Sulfur Dioxide", "SO₂", 1.63f)); // Sulfur Dioxide
        dipoleMomentMolecules.Add(new DipoleMomentMolecule("Nitrogen Monoxide", "NO", 0.16f)); // Nitrogen Monoxide
    }
}
