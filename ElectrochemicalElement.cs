using UnityEngine;

[System.Serializable]
public class ElectrochemicalElement: MonoBehaviour
{
    public string elementName;  // Name of the element
    public float reductionPotential;  // Standard reduction potential in volts (V)

    public ElectrochemicalElement(string elementName, float reductionPotential)
    {
        this.elementName = elementName;
        this.reductionPotential = reductionPotential;
    }
}
