using System.Collections.Generic;
using UnityEngine;

public class EmissionSpectrumGenerator : MonoBehaviour
{
    [Header("Emission Spectrum of Elements")]
    public List<EmissionLine> emissionLines = new List<EmissionLine>();

    void Start()
    {
        GenerateEmissionSpectrum();
    }

    [ContextMenu("Generate Emission Spectrum")]
    public void GenerateEmissionSpectrum()
    {
        emissionLines.Clear();  // Clear the list before generating

        // Define custom colors
        Color violet = new Color(0.56f, 0.0f, 1.0f);    // Violet (RGB: R: 0.56, G: 0.0, B: 1.0)
        Color orange = new Color(1.0f, 0.65f, 0.0f);    // Orange (RGB: R: 1.0, G: 0.65, B: 0.0)

        // Add common elements and their emission lines (wavelength in nm, color based on wavelength)
        
        // Hydrogen Emission Spectrum (Balmer Series)
        emissionLines.Add(new EmissionLine("Hydrogen", 410.2f, Color.magenta));  // Violet
        emissionLines.Add(new EmissionLine("Hydrogen", 434.0f, Color.blue));     // Blue
        emissionLines.Add(new EmissionLine("Hydrogen", 486.1f, Color.cyan));     // Cyan
        emissionLines.Add(new EmissionLine("Hydrogen", 656.3f, Color.red));      // Red

        // Helium Emission Spectrum
        emissionLines.Add(new EmissionLine("Helium", 447.1f, Color.blue));       // Blue
        emissionLines.Add(new EmissionLine("Helium", 471.3f, Color.cyan));       // Cyan
        emissionLines.Add(new EmissionLine("Helium", 492.2f, Color.green));      // Green
        emissionLines.Add(new EmissionLine("Helium", 501.6f, Color.green));      // Green
        emissionLines.Add(new EmissionLine("Helium", 587.6f, Color.yellow));     // Yellow
        emissionLines.Add(new EmissionLine("Helium", 667.8f, Color.red));        // Red
        emissionLines.Add(new EmissionLine("Helium", 706.5f, Color.red));        // Red

        // Sodium Emission Spectrum (Na D-line)
        emissionLines.Add(new EmissionLine("Sodium", 589.0f, Color.yellow));     // Yellow

        // Neon Emission Spectrum
        emissionLines.Add(new EmissionLine("Neon", 540.1f, Color.green));        // Green
        emissionLines.Add(new EmissionLine("Neon", 585.2f, Color.yellow));       // Yellow
        emissionLines.Add(new EmissionLine("Neon", 640.2f, Color.red));          // Red
        emissionLines.Add(new EmissionLine("Neon", 703.2f, Color.red));          // Deep Red
        emissionLines.Add(new EmissionLine("Neon", 743.9f, Color.red));          // Deep Red
        emissionLines.Add(new EmissionLine("Neon", 837.8f, Color.red));          // Near Infrared

        // Mercury Emission Spectrum
        emissionLines.Add(new EmissionLine("Mercury", 404.7f, violet));          // Violet
        emissionLines.Add(new EmissionLine("Mercury", 435.8f, Color.blue));      // Blue
        emissionLines.Add(new EmissionLine("Mercury", 546.1f, Color.green));     // Green
        emissionLines.Add(new EmissionLine("Mercury", 577.0f, Color.yellow));    // Yellow
        emissionLines.Add(new EmissionLine("Mercury", 579.1f, Color.yellow));    // Yellow

        // Argon Emission Spectrum
        emissionLines.Add(new EmissionLine("Argon", 696.5f, Color.red));         // Red
        emissionLines.Add(new EmissionLine("Argon", 763.5f, Color.red));         // Red
        emissionLines.Add(new EmissionLine("Argon", 811.5f, Color.red));         // Red

        // Krypton Emission Spectrum
        emissionLines.Add(new EmissionLine("Krypton", 427.4f, violet));          // Violet
        emissionLines.Add(new EmissionLine("Krypton", 557.0f, Color.green));     // Green
        emissionLines.Add(new EmissionLine("Krypton", 587.1f, Color.yellow));    // Yellow
        emissionLines.Add(new EmissionLine("Krypton", 760.2f, Color.red));       // Red

        // Oxygen Emission Spectrum
        emissionLines.Add(new EmissionLine("Oxygen", 557.7f, Color.green));      // Green
        emissionLines.Add(new EmissionLine("Oxygen", 630.0f, Color.red));        // Red

        // Nitrogen Emission Spectrum
        emissionLines.Add(new EmissionLine("Nitrogen", 337.1f, violet));         // Violet
        emissionLines.Add(new EmissionLine("Nitrogen", 357.7f, violet));         // Violet
        emissionLines.Add(new EmissionLine("Nitrogen", 500.5f, Color.green));    // Green

        // Chlorine Emission Spectrum
        emissionLines.Add(new EmissionLine("Chlorine", 551.0f, Color.green));    // Green
        emissionLines.Add(new EmissionLine("Chlorine", 579.1f, Color.yellow));   // Yellow
        emissionLines.Add(new EmissionLine("Chlorine", 620.0f, orange));         // Orange
        emissionLines.Add(new EmissionLine("Chlorine", 682.0f, Color.red));      // Red
    }
}
