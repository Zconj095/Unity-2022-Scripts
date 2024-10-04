using UnityEngine;

[System.Serializable]
public class EmissionLine
{
    public string elementName;      // Name of the element (e.g., "Hydrogen")
    public float wavelength;        // Wavelength in nanometers (nm)
    public Color emissionColor;     // Color of the emission line (based on wavelength)

    public EmissionLine(string elementName, float wavelength, Color emissionColor)
    {
        this.elementName = elementName;
        this.wavelength = wavelength;
        this.emissionColor = emissionColor;
    }
}
