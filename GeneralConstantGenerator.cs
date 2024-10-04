using System.Collections.Generic;
using UnityEngine;

public class GeneralConstantGenerator : MonoBehaviour
{
    [Header("List of General Constants")]
    public List<GeneralConstant> constants = new List<GeneralConstant>();

    void Start()
    {
        GenerateGeneralConstants();
    }

    [ContextMenu("Generate General Constants")]
    public void GenerateGeneralConstants()
    {
        constants.Clear();  // Clear the list before generating

        // Add general constants: name, symbol, value, and units
        constants.Add(new GeneralConstant("Speed of Light", "c", 299792458f, "m/s"));  // Speed of light
        constants.Add(new GeneralConstant("Gravitational Constant", "G", 6.67430e-11f, "m³ kg⁻¹ s⁻²"));  // Gravitational constant
        constants.Add(new GeneralConstant("Planck's Constant", "h", 6.62607015e-34f, "J·s"));  // Planck's constant
        constants.Add(new GeneralConstant("Elementary Charge", "e", 1.602176634e-19f, "C"));  // Elementary charge
        constants.Add(new GeneralConstant("Boltzmann Constant", "k", 1.380649e-23f, "J/K"));  // Boltzmann constant
        constants.Add(new GeneralConstant("Avogadro's Number", "N_A", 6.02214076e23f, "mol⁻¹"));  // Avogadro's number
        constants.Add(new GeneralConstant("Gas Constant", "R", 8.314462618f, "J/(mol·K)"));  // Gas constant
        constants.Add(new GeneralConstant("Stefan-Boltzmann Constant", "σ", 5.670374419e-8f, "W/(m²·K⁴)"));  // Stefan-Boltzmann constant
        constants.Add(new GeneralConstant("Electron Mass", "m_e", 9.10938356e-31f, "kg"));  // Electron mass
        constants.Add(new GeneralConstant("Proton Mass", "m_p", 1.67262192369e-27f, "kg"));  // Proton mass
        constants.Add(new GeneralConstant("Neutron Mass", "m_n", 1.67492749804e-27f, "kg"));  // Neutron mass
        constants.Add(new GeneralConstant("Permittivity of Free Space", "ε₀", 8.854187817e-12f, "F/m"));  // Electric constant
        constants.Add(new GeneralConstant("Permeability of Free Space", "μ₀", 1.2566370614e-6f, "N/A²"));  // Magnetic constant
        constants.Add(new GeneralConstant("Fine Structure Constant", "α", 7.2973525693e-3f, "dimensionless"));  // Fine-structure constant
        constants.Add(new GeneralConstant("Molar Mass Constant", "Mₐ", 1.00000000e-3f, "kg/mol"));  // Molar mass constant
    }
}
