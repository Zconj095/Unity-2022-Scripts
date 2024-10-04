using UnityEngine;

[System.Serializable]
public class GeneralConstant: MonoBehaviour
{
    public string constantName;  // Name of the constant (e.g., "Speed of Light")
    public string symbol;        // Symbol of the constant (e.g., "c")
    public float value;          // Value of the constant (e.g., 299792458 for speed of light)
    public string units;         // Units of the constant (e.g., "m/s")

    public GeneralConstant(string constantName, string symbol, float value, string units)
    {
        this.constantName = constantName;
        this.symbol = symbol;
        this.value = value;
        this.units = units;
    }
}
