using UnityEngine;

[System.Serializable]
public class Element: MonoBehaviour
{
    public int atomicNumber;
    public string elementName;
    public string symbol;
    public float atomicMass;

    public Element(int atomicNumber, string elementName, string symbol, float atomicMass)
    {
        this.atomicNumber = atomicNumber;
        this.elementName = elementName;
        this.symbol = symbol;
        this.atomicMass = atomicMass;
    }
}
