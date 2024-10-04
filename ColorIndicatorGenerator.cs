using System.Collections.Generic;
using UnityEngine;

public class ColorIndicatorGenerator : MonoBehaviour
{
    [Header("Color Indicators")]
    public List<ColorIndicator> indicators = new List<ColorIndicator>();  // List to store pH indicators

    void Start()
    {
        GenerateColorIndicators();
    }

    [ContextMenu("Generate Color Indicators")]
    public void GenerateColorIndicators()
    {
        indicators.Clear();  // Clear the list before generating

        // Add common pH indicators, their color changes, and pH transition ranges
        indicators.Add(new ColorIndicator("Phenolphthalein", 8.2f, 10.0f, Color.clear, Color.magenta));  // Clear to magenta
        indicators.Add(new ColorIndicator("Methyl Orange", 3.1f, 4.4f, Color.red, Color.yellow));  // Red to yellow
        indicators.Add(new ColorIndicator("Bromothymol Blue", 6.0f, 7.6f, Color.yellow, Color.blue));  // Yellow to blue
        indicators.Add(new ColorIndicator("Litmus", 4.5f, 8.3f, Color.red, Color.blue));  // Red to blue
        indicators.Add(new ColorIndicator("Thymol Blue", 1.2f, 2.8f, Color.red, Color.yellow));  // Red to yellow (1st range)
        indicators.Add(new ColorIndicator("Thymol Blue", 8.0f, 9.6f, Color.yellow, Color.blue));  // Yellow to blue (2nd range)
        indicators.Add(new ColorIndicator("Universal Indicator", 4.0f, 10.0f, new Color(1f, 0.2f, 0.2f), new Color(0.2f, 1f, 0.2f)));  // Red to green
        indicators.Add(new ColorIndicator("Cresol Red", 7.2f, 8.8f, Color.yellow, Color.red));  // Yellow to red
        indicators.Add(new ColorIndicator("Methyl Red", 4.4f, 6.2f, Color.red, Color.yellow));  // Red to yellow
    }
}
