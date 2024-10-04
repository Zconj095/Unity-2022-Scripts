using System.Collections.Generic;
using UnityEngine;

public class IonizationEnergyGenerator : MonoBehaviour
{
    [Header("Ionization Energy of Elements")]
    public List<IonizationEnergyElement> ionizationEnergies = new List<IonizationEnergyElement>();  // List to store ionization energies

    void Start()
    {
        GenerateIonizationEnergies();
    }

    [ContextMenu("Generate Ionization Energies")]
    public void GenerateIonizationEnergies()
    {
        ionizationEnergies.Clear();  // Clear the list before generating

        // Add common elements and their first ionization energies (in electron volts eV)
        ionizationEnergies.Add(new IonizationEnergyElement("Hydrogen (H)", 1, 13.6f));  // Ionization energy in eV
        ionizationEnergies.Add(new IonizationEnergyElement("Helium (He)", 2, 24.6f));
        ionizationEnergies.Add(new IonizationEnergyElement("Lithium (Li)", 3, 5.39f));
        ionizationEnergies.Add(new IonizationEnergyElement("Beryllium (Be)", 4, 9.32f));
        ionizationEnergies.Add(new IonizationEnergyElement("Boron (B)", 5, 8.30f));
        ionizationEnergies.Add(new IonizationEnergyElement("Carbon (C)", 6, 11.26f));
        ionizationEnergies.Add(new IonizationEnergyElement("Nitrogen (N)", 7, 14.53f));
        ionizationEnergies.Add(new IonizationEnergyElement("Oxygen (O)", 8, 13.62f));
        ionizationEnergies.Add(new IonizationEnergyElement("Fluorine (F)", 9, 17.42f));
        ionizationEnergies.Add(new IonizationEnergyElement("Neon (Ne)", 10, 21.56f));
        ionizationEnergies.Add(new IonizationEnergyElement("Sodium (Na)", 11, 5.14f));
        ionizationEnergies.Add(new IonizationEnergyElement("Magnesium (Mg)", 12, 7.65f));
        ionizationEnergies.Add(new IonizationEnergyElement("Aluminum (Al)", 13, 5.99f));
        ionizationEnergies.Add(new IonizationEnergyElement("Silicon (Si)", 14, 8.15f));
        ionizationEnergies.Add(new IonizationEnergyElement("Phosphorus (P)", 15, 10.49f));
        ionizationEnergies.Add(new IonizationEnergyElement("Sulfur (S)", 16, 10.36f));
        ionizationEnergies.Add(new IonizationEnergyElement("Chlorine (Cl)", 17, 12.97f));
        ionizationEnergies.Add(new IonizationEnergyElement("Argon (Ar)", 18, 15.76f));
        ionizationEnergies.Add(new IonizationEnergyElement("Potassium (K)", 19, 4.34f));
        ionizationEnergies.Add(new IonizationEnergyElement("Calcium (Ca)", 20, 6.11f));
        ionizationEnergies.Add(new IonizationEnergyElement("Hydrogen (H)", 1, 13.6f));  // Ionization energy in eV
        ionizationEnergies.Add(new IonizationEnergyElement("Helium (He)", 2, 24.6f));
        ionizationEnergies.Add(new IonizationEnergyElement("Lithium (Li)", 3, 5.39f));
        ionizationEnergies.Add(new IonizationEnergyElement("Beryllium (Be)", 4, 9.32f));
        ionizationEnergies.Add(new IonizationEnergyElement("Boron (B)", 5, 8.30f));
        ionizationEnergies.Add(new IonizationEnergyElement("Carbon (C)", 6, 11.26f));
        ionizationEnergies.Add(new IonizationEnergyElement("Nitrogen (N)", 7, 14.53f));
        ionizationEnergies.Add(new IonizationEnergyElement("Oxygen (O)", 8, 13.62f));
        ionizationEnergies.Add(new IonizationEnergyElement("Fluorine (F)", 9, 17.42f));
        ionizationEnergies.Add(new IonizationEnergyElement("Neon (Ne)", 10, 21.56f));
        ionizationEnergies.Add(new IonizationEnergyElement("Sodium (Na)", 11, 5.14f));
        ionizationEnergies.Add(new IonizationEnergyElement("Magnesium (Mg)", 12, 7.65f));
        ionizationEnergies.Add(new IonizationEnergyElement("Aluminum (Al)", 13, 5.99f));
        ionizationEnergies.Add(new IonizationEnergyElement("Silicon (Si)", 14, 8.15f));
        ionizationEnergies.Add(new IonizationEnergyElement("Phosphorus (P)", 15, 10.49f));
        ionizationEnergies.Add(new IonizationEnergyElement("Sulfur (S)", 16, 10.36f));
        ionizationEnergies.Add(new IonizationEnergyElement("Chlorine (Cl)", 17, 12.97f));
        ionizationEnergies.Add(new IonizationEnergyElement("Argon (Ar)", 18, 15.76f));
        ionizationEnergies.Add(new IonizationEnergyElement("Potassium (K)", 19, 4.34f));
        ionizationEnergies.Add(new IonizationEnergyElement("Calcium (Ca)", 20, 6.11f));

        // Continuing with other common elements
        ionizationEnergies.Add(new IonizationEnergyElement("Scandium (Sc)", 21, 6.56f));
        ionizationEnergies.Add(new IonizationEnergyElement("Titanium (Ti)", 22, 6.83f));
        ionizationEnergies.Add(new IonizationEnergyElement("Vanadium (V)", 23, 6.75f));
        ionizationEnergies.Add(new IonizationEnergyElement("Chromium (Cr)", 24, 6.77f));
        ionizationEnergies.Add(new IonizationEnergyElement("Manganese (Mn)", 25, 7.43f));
        ionizationEnergies.Add(new IonizationEnergyElement("Iron (Fe)", 26, 7.87f));
        ionizationEnergies.Add(new IonizationEnergyElement("Cobalt (Co)", 27, 7.86f));
        ionizationEnergies.Add(new IonizationEnergyElement("Nickel (Ni)", 28, 7.64f));
        ionizationEnergies.Add(new IonizationEnergyElement("Copper (Cu)", 29, 7.72f));
        ionizationEnergies.Add(new IonizationEnergyElement("Zinc (Zn)", 30, 9.39f));

        // You can continue adding more elements as needed.

    }
}
