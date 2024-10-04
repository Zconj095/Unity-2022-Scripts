using UnityEngine;

[System.Serializable]
public class ElectronegativityElement: MonoBehaviour
{
    public string elementSymbol;     // Element symbol (e.g., "H", "O", "Cl")
    public int atomicNumber;         // Atomic number (e.g., 1 for Hydrogen)
    public float electronegativity;  // Electronegativity on the Pauling scale

    public ElectronegativityElement(string elementSymbol, int atomicNumber, float electronegativity)
    {
        this.elementSymbol = elementSymbol;
        this.atomicNumber = atomicNumber;
        this.electronegativity = electronegativity;
    }
}
