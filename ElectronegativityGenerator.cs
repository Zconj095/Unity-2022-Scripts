using System.Collections.Generic;
using UnityEngine;

public class ElectronegativityGenerator : MonoBehaviour
{
    [Header("Electronegativity Values")]
    public List<ElectronegativityElement> electronegativityElements = new List<ElectronegativityElement>();

    void Start()
    {
        GenerateElectronegativityValues();
    }

    [ContextMenu("Generate Electronegativity Values")]
    public void GenerateElectronegativityValues()
    {
        electronegativityElements.Clear();  // Clear the list before generating

        // Add common elements and their electronegativity values (Pauling Scale)
        // Period 1
        electronegativityElements.Add(new ElectronegativityElement("H", 1, 2.20f));  // Hydrogen
        electronegativityElements.Add(new ElectronegativityElement("He", 2, 0.00f)); // Helium (inert)

        // Period 2
        electronegativityElements.Add(new ElectronegativityElement("Li", 3, 0.98f));  // Lithium
        electronegativityElements.Add(new ElectronegativityElement("Be", 4, 1.57f));  // Beryllium
        electronegativityElements.Add(new ElectronegativityElement("B", 5, 2.04f));   // Boron
        electronegativityElements.Add(new ElectronegativityElement("C", 6, 2.55f));   // Carbon
        electronegativityElements.Add(new ElectronegativityElement("N", 7, 3.04f));   // Nitrogen
        electronegativityElements.Add(new ElectronegativityElement("O", 8, 3.44f));   // Oxygen
        electronegativityElements.Add(new ElectronegativityElement("F", 9, 3.98f));   // Fluorine (highest electronegativity)
        electronegativityElements.Add(new ElectronegativityElement("Ne", 10, 0.00f)); // Neon (inert)

        // Period 3
        electronegativityElements.Add(new ElectronegativityElement("Na", 11, 0.93f)); // Sodium
        electronegativityElements.Add(new ElectronegativityElement("Mg", 12, 1.31f)); // Magnesium
        electronegativityElements.Add(new ElectronegativityElement("Al", 13, 1.61f)); // Aluminum
        electronegativityElements.Add(new ElectronegativityElement("Si", 14, 1.90f)); // Silicon
        electronegativityElements.Add(new ElectronegativityElement("P", 15, 2.19f));  // Phosphorus
        electronegativityElements.Add(new ElectronegativityElement("S", 16, 2.58f));  // Sulfur
        electronegativityElements.Add(new ElectronegativityElement("Cl", 17, 3.16f)); // Chlorine
        electronegativityElements.Add(new ElectronegativityElement("Ar", 18, 0.00f)); // Argon (inert)

        // Period 4
        electronegativityElements.Add(new ElectronegativityElement("K", 19, 0.82f));  // Potassium
        electronegativityElements.Add(new ElectronegativityElement("Ca", 20, 1.00f)); // Calcium
        electronegativityElements.Add(new ElectronegativityElement("Sc", 21, 1.36f)); // Scandium
        electronegativityElements.Add(new ElectronegativityElement("Ti", 22, 1.54f)); // Titanium
        electronegativityElements.Add(new ElectronegativityElement("V", 23, 1.63f));  // Vanadium
        electronegativityElements.Add(new ElectronegativityElement("Cr", 24, 1.66f)); // Chromium
        electronegativityElements.Add(new ElectronegativityElement("Mn", 25, 1.55f)); // Manganese
        electronegativityElements.Add(new ElectronegativityElement("Fe", 26, 1.83f)); // Iron
        electronegativityElements.Add(new ElectronegativityElement("Co", 27, 1.88f)); // Cobalt
        electronegativityElements.Add(new ElectronegativityElement("Ni", 28, 1.91f)); // Nickel
        electronegativityElements.Add(new ElectronegativityElement("Cu", 29, 1.90f)); // Copper
        electronegativityElements.Add(new ElectronegativityElement("Zn", 30, 1.65f)); // Zinc
        electronegativityElements.Add(new ElectronegativityElement("Ga", 31, 1.81f)); // Gallium
        electronegativityElements.Add(new ElectronegativityElement("Ge", 32, 2.01f)); // Germanium
        electronegativityElements.Add(new ElectronegativityElement("As", 33, 2.18f)); // Arsenic
        electronegativityElements.Add(new ElectronegativityElement("Se", 34, 2.55f)); // Selenium
        electronegativityElements.Add(new ElectronegativityElement("Br", 35, 2.96f)); // Bromine
        electronegativityElements.Add(new ElectronegativityElement("Kr", 36, 3.00f)); // Krypton

        // Period 5
        electronegativityElements.Add(new ElectronegativityElement("Rb", 37, 0.82f)); // Rubidium
        electronegativityElements.Add(new ElectronegativityElement("Sr", 38, 0.95f)); // Strontium
        electronegativityElements.Add(new ElectronegativityElement("Y", 39, 1.22f));  // Yttrium
        electronegativityElements.Add(new ElectronegativityElement("Zr", 40, 1.33f)); // Zirconium
        electronegativityElements.Add(new ElectronegativityElement("Nb", 41, 1.6f));  // Niobium
        electronegativityElements.Add(new ElectronegativityElement("Mo", 42, 2.16f)); // Molybdenum
        electronegativityElements.Add(new ElectronegativityElement("Tc", 43, 1.9f));  // Technetium
        electronegativityElements.Add(new ElectronegativityElement("Ru", 44, 2.2f));  // Ruthenium
        electronegativityElements.Add(new ElectronegativityElement("Rh", 45, 2.28f)); // Rhodium
        electronegativityElements.Add(new ElectronegativityElement("Pd", 46, 2.20f)); // Palladium
        electronegativityElements.Add(new ElectronegativityElement("Ag", 47, 1.93f)); // Silver
        electronegativityElements.Add(new ElectronegativityElement("Cd", 48, 1.69f)); // Cadmium
        electronegativityElements.Add(new ElectronegativityElement("In", 49, 1.78f)); // Indium
        electronegativityElements.Add(new ElectronegativityElement("Sn", 50, 1.96f)); // Tin
        electronegativityElements.Add(new ElectronegativityElement("Sb", 51, 2.05f)); // Antimony
        electronegativityElements.Add(new ElectronegativityElement("Te", 52, 2.1f));  // Tellurium
        electronegativityElements.Add(new ElectronegativityElement("I", 53, 2.66f));  // Iodine
        electronegativityElements.Add(new ElectronegativityElement("Xe", 54, 2.60f)); // Xenon

        // Period 6
        electronegativityElements.Add(new ElectronegativityElement("Cs", 55, 0.79f)); // Cesium
        electronegativityElements.Add(new ElectronegativityElement("Ba", 56, 0.89f)); // Barium
        electronegativityElements.Add(new ElectronegativityElement("La", 57, 1.10f)); // Lanthanum
        electronegativityElements.Add(new ElectronegativityElement("Ce", 58, 1.12f)); // Cerium
        electronegativityElements.Add(new ElectronegativityElement("Pr", 59, 1.13f)); // Praseodymium
        electronegativityElements.Add(new ElectronegativityElement("Nd", 60, 1.14f)); // Neodymium
        electronegativityElements.Add(new ElectronegativityElement("Pm", 61, 1.13f)); // Promethium
        electronegativityElements.Add(new ElectronegativityElement("Sm", 62, 1.17f)); // Samarium
        electronegativityElements.Add(new ElectronegativityElement("Eu", 63, 1.2f));  // Europium
        electronegativityElements.Add(new ElectronegativityElement("Gd", 64, 1.2f));  // Gadolinium
        electronegativityElements.Add(new ElectronegativityElement("Tb", 65, 1.1f));  // Terbium
        electronegativityElements.Add(new ElectronegativityElement("Dy", 66, 1.22f)); // Dysprosium
        electronegativityElements.Add(new ElectronegativityElement("Ho", 67, 1.23f)); // Holmium
        electronegativityElements.Add(new ElectronegativityElement("Er", 68, 1.24f)); // Erbium
        electronegativityElements.Add(new ElectronegativityElement("Tm", 69, 1.25f)); // Thulium
        electronegativityElements.Add(new ElectronegativityElement("Yb", 70, 1.1f));  // Ytterbium
        electronegativityElements.Add(new ElectronegativityElement("Lu", 71, 1.27f)); // Lutetium
        electronegativityElements.Add(new ElectronegativityElement("Hf", 72, 1.3f));  // Hafnium
        electronegativityElements.Add(new ElectronegativityElement("Ta", 73, 1.5f));  // Tantalum
        electronegativityElements.Add(new ElectronegativityElement("W", 74, 2.36f));  // Tungsten
        electronegativityElements.Add(new ElectronegativityElement("Re", 75, 1.9f));  // Rhenium
        electronegativityElements.Add(new ElectronegativityElement("Os", 76, 2.2f));  // Osmium
        electronegativityElements.Add(new ElectronegativityElement("Ir", 77, 2.2f));  // Iridium
        electronegativityElements.Add(new ElectronegativityElement("Pt", 78, 2.28f)); // Platinum
        electronegativityElements.Add(new ElectronegativityElement("Au", 79, 2.54f)); // Gold
        electronegativityElements.Add(new ElectronegativityElement("Hg", 80, 2.00f)); // Mercury
        electronegativityElements.Add(new ElectronegativityElement("Tl", 81, 1.62f)); // Thallium
        electronegativityElements.Add(new ElectronegativityElement("Pb", 82, 2.33f)); // Lead
        electronegativityElements.Add(new ElectronegativityElement("Bi", 83, 2.02f)); // Bismuth
        electronegativityElements.Add(new ElectronegativityElement("Po", 84, 2.00f)); // Polonium
        electronegativityElements.Add(new ElectronegativityElement("At", 85, 2.2f));  // Astatine
        electronegativityElements.Add(new ElectronegativityElement("Rn", 86, 0.00f)); // Radon (inert)

    }
}
