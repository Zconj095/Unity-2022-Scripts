using System.Collections.Generic;
using UnityEngine;

public class AlkaneGenerator : MonoBehaviour
{
    [Header("Properties of Alkanes")]
    public List<Alkane> alkanes = new List<Alkane>();  // List to store all alkanes

    void Start()
    {
        GenerateAlkanes();
    }

    [ContextMenu("Generate Alkanes")]
    public void GenerateAlkanes()
    {
        alkanes.Clear();  // Clear the list before generating

        // Add common alkanes and their properties (molecular formula, molecular weight, boiling point, melting point, density)
        alkanes.Add(new Alkane("Methane", "CH₄", 16.04f, -161.5f, -182.5f, 0.656f));  // Density in g/L
        alkanes.Add(new Alkane("Ethane", "C₂H₆", 30.07f, -88.6f, -183.3f, 1.356f));  // Density in g/L
        alkanes.Add(new Alkane("Propane", "C₃H₈", 44.10f, -42.1f, -187.7f, 1.867f));  // Density in g/L
        alkanes.Add(new Alkane("Butane", "C₄H₁₀", 58.12f, -0.5f, -138.4f, 2.493f));  // Density in g/L
        alkanes.Add(new Alkane("Pentane", "C₅H₁₂", 72.15f, 36.1f, -129.7f, 0.626f));  // Density in g/cm³
        alkanes.Add(new Alkane("Hexane", "C₆H₁₄", 86.18f, 68.7f, -95.3f, 0.655f));  // Density in g/cm³
        alkanes.Add(new Alkane("Heptane", "C₇H₁₆", 100.20f, 98.4f, -90.6f, 0.684f));  // Density in g/cm³
        alkanes.Add(new Alkane("Octane", "C₈H₁₈", 114.23f, 125.6f, -56.8f, 0.703f));  // Density in g/cm³
        alkanes.Add(new Alkane("Nonane", "C₉H₂₀", 128.26f, 150.8f, -53.5f, 0.718f));  // Density in g/cm³
        alkanes.Add(new Alkane("Decane", "C₁₀H₂₂", 142.29f, 174.1f, -29.7f, 0.730f));  // Density in g/cm³

        // Continuing with larger alkanes
        alkanes.Add(new Alkane("Undecane", "C₁₁H₂₄", 156.32f, 195.9f, -25.6f, 0.740f));  // Density in g/cm³
        alkanes.Add(new Alkane("Dodecane", "C₁₂H₂₆", 170.33f, 216.3f, -9.6f, 0.749f));  // Density in g/cm³
        alkanes.Add(new Alkane("Tridecane", "C₁₃H₂₈", 184.36f, 234.7f, -5.2f, 0.756f));  // Density in g/cm³
        alkanes.Add(new Alkane("Tetradecane", "C₁₄H₃₀", 198.39f, 253.5f, 5.9f, 0.763f));  // Density in g/cm³
        alkanes.Add(new Alkane("Pentadecane", "C₁₅H₃₂", 212.42f, 270.8f, 9.9f, 0.769f));  // Density in g/cm³
        alkanes.Add(new Alkane("Hexadecane", "C₁₆H₃₄", 226.44f, 287.0f, 18.0f, 0.773f));  // Density in g/cm³

    }
}
