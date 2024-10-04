using UnityEngine;

[System.Serializable]
public class ColorIndicator: MonoBehaviour
{
    public string indicatorName;  // Name of the indicator (e.g., "Phenolphthalein")
    public float lowerPH;         // Lower pH transition value
    public float upperPH;         // Upper pH transition value
    public Color acidicColor;     // Color at acidic pH
    public Color basicColor;      // Color at basic pH

    public ColorIndicator(string indicatorName, float lowerPH, float upperPH, Color acidicColor, Color basicColor)
    {
        this.indicatorName = indicatorName;
        this.lowerPH = lowerPH;
        this.upperPH = upperPH;
        this.acidicColor = acidicColor;
        this.basicColor = basicColor;
    }
}
