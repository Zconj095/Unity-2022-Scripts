using System.Collections.Generic;
using UnityEngine;

public class NeutronCrossSectionGenerator : MonoBehaviour
{
    [Header("Neutron Cross-Sections of Isotopes")]
    public List<NeutronCrossSection> neutronCrossSections = new List<NeutronCrossSection>();

    void Start()
    {
        GenerateNeutronCrossSections();
    }

    [ContextMenu("Generate Neutron Cross-Sections")]
    public void GenerateNeutronCrossSections()
    {
        neutronCrossSections.Clear();  // Clear the list before generating
        neutronCrossSections.Add(new NeutronCrossSection("Uranium-235", 680.0f, 11.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Uranium-238", 2.68f, 8.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Hydrogen-1", 0.332f, 20.4f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Deuterium", 0.000519f, 4.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Boron-10", 3837.0f, 3.7f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Carbon-12", 0.0035f, 5.5f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Xenon-135", 2.65e6f, 3.3f, "Absorption"));
        neutronCrossSections.Add(new NeutronCrossSection("Plutonium-239", 1013.0f, 10.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Helium-3", 5330.0f, 1.7f, "Absorption and Scattering"));

        // Continuing with additional isotopes:
        neutronCrossSections.Add(new NeutronCrossSection("Beryllium-9", 0.0092f, 7.6f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Lead-208", 0.171f, 11.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Iron-56", 2.55f, 11.22f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Cadmium-113", 20600.0f, 7.5f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Thorium-232", 7.4f, 11.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Lithium-6", 940.0f, 0.85f, "Absorption"));
        neutronCrossSections.Add(new NeutronCrossSection("Oxygen-16", 0.00019f, 3.76f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Nickel-58", 4.6f, 10.3f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Silver-107", 37.6f, 8.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Copper-63", 4.5f, 8.0f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Aluminum-27", 0.231f, 1.49f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Zirconium-91", 1.24f, 7.4f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Silicon-28", 0.16f, 2.2f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Chromium-52", 3.07f, 4.6f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Tungsten-184", 1.7f, 5.4f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Gold-197", 98.65f, 9.2f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Cobalt-59", 37.18f, 4.9f, "Absorption and Scattering"));
        neutronCrossSections.Add(new NeutronCrossSection("Europium-151", 4300.0f, 8.6f, "Absorption and Scattering"));

    }
}
