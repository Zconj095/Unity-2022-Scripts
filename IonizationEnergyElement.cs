using UnityEngine;

[System.Serializable]
public class IonizationEnergyElement: MonoBehaviour
{
    public string elementName;  // Name of the element (e.g., "Hydrogen")
    public int atomicNumber;    // Atomic number of the element
    public float ionizationEnergy;  // First ionization energy in electron volts (eV) or kilojoules per mole (kJ/mol)

    public IonizationEnergyElement(string elementName, int atomicNumber, float ionizationEnergy)
    {
        this.elementName = elementName;
        this.atomicNumber = atomicNumber;
        this.ionizationEnergy = ionizationEnergy;
    }
}
