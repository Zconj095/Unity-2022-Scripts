using UnityEngine;

[System.Serializable]
public class ElectronSubshell: MonoBehaviour
{
    public int energyLevel;  // Principal energy level (n)
    public char orbitalType; // s, p, d, f
    public int maxElectrons; // Maximum number of electrons in the orbital

    public ElectronSubshell(int energyLevel, char orbitalType, int maxElectrons)
    {
        this.energyLevel = energyLevel;
        this.orbitalType = orbitalType;
        this.maxElectrons = maxElectrons;
    }
}
