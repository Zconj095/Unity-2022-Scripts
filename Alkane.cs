using UnityEngine;

[System.Serializable]
public class Alkane: MonoBehaviour
{
    public string alkaneName;        // Name of the alkane (e.g., "Methane")
    public string molecularFormula;  // Molecular formula (e.g., "CH₄")
    public float molecularWeight;    // Molecular weight in g/mol
    public float boilingPoint;       // Boiling point in °C
    public float meltingPoint;       // Melting point in °C
    public float density;            // Density in g/cm³ (or kg/m³)

    public Alkane(string alkaneName, string molecularFormula, float molecularWeight, float boilingPoint, float meltingPoint, float density)
    {
        this.alkaneName = alkaneName;
        this.molecularFormula = molecularFormula;
        this.molecularWeight = molecularWeight;
        this.boilingPoint = boilingPoint;
        this.meltingPoint = meltingPoint;
        this.density = density;
    }
}
