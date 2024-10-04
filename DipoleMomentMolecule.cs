using UnityEngine;

[System.Serializable]
public class DipoleMomentMolecule: MonoBehaviour
{
    public string moleculeName;     // Name of the molecule (e.g., "Water")
    public string chemicalFormula;  // Chemical formula of the molecule (e.g., "H₂O")
    public float dipoleMoment;      // Dipole moment value in Debye (D)

    public DipoleMomentMolecule(string moleculeName, string chemicalFormula, float dipoleMoment)
    {
        this.moleculeName = moleculeName;
        this.chemicalFormula = chemicalFormula;
        this.dipoleMoment = dipoleMoment;
    }
}
