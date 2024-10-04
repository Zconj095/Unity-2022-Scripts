using UnityEngine;

[System.Serializable]
public class PolyaromaticCarbon: MonoBehaviour
{
    public string name;  // Name of the polyaromatic hydrocarbon (e.g., "Benzene")
    public string molecularFormula;  // Molecular formula (e.g., "C6H6")
    public float molecularWeight;  // Molecular weight in g/mol
    public int numberOfRings;  // Number of aromatic rings
    public float meltingPoint;  // Melting point in °C
    public float boilingPoint;  // Boiling point in °C

    public PolyaromaticCarbon(string name, string molecularFormula, float molecularWeight, int numberOfRings, float meltingPoint, float boilingPoint)
    {
        this.name = name;
        this.molecularFormula = molecularFormula;
        this.molecularWeight = molecularWeight;
        this.numberOfRings = numberOfRings;
        this.meltingPoint = meltingPoint;
        this.boilingPoint = boilingPoint;
    }
}
