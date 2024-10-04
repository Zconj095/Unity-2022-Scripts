using System.Collections.Generic;
using System;
using UnityEngine;

public class SolubilityTableGenerator : MonoBehaviour
{
    [Header("Solubility Table")]
    public List<CompoundSolubility> compounds = new List<CompoundSolubility>();

    void Start()
    {
        GenerateSolubilityTable();
    }

    [ContextMenu("Generate Solubility Table")]
    public void GenerateSolubilityTable()
    {
        compounds.Clear(); // Clear the list before generating

        // Add common ionic compounds and their solubility
        compounds.Add(new CompoundSolubility("Sodium Chloride", "NaCl", "Soluble"));
        compounds.Add(new CompoundSolubility("Silver Chloride", "AgCl", "Insoluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Chloride", "PbCl2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Nitrate", "KNO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Carbonate", "CaCO3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Barium Sulfate", "BaSO4", "Insoluble"));
        compounds.Add(new CompoundSolubility("Magnesium Hydroxide", "Mg(OH)2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Copper(II) Sulfate", "CuSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Ammonium Nitrate", "NH4NO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Hydroxide", "Ca(OH)2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Iron(III) Hydroxide", "Fe(OH)3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Zinc Sulfate", "ZnSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Sodium Chloride", "NaCl", "Soluble"));
        compounds.Add(new CompoundSolubility("Silver Chloride", "AgCl", "Insoluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Chloride", "PbCl2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Nitrate", "KNO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Carbonate", "CaCO3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Barium Sulfate", "BaSO4", "Insoluble"));
        compounds.Add(new CompoundSolubility("Magnesium Hydroxide", "Mg(OH)2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Copper(II) Sulfate", "CuSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Ammonium Nitrate", "NH4NO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Hydroxide", "Ca(OH)2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Iron(III) Hydroxide", "Fe(OH)3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Zinc Sulfate", "ZnSO4", "Soluble"));

        // Continue adding more common ionic compounds
        compounds.Add(new CompoundSolubility("Sodium Carbonate", "Na2CO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Silver Bromide", "AgBr", "Insoluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Bromide", "PbBr2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Sodium Hydroxide", "NaOH", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Sulfate", "CaSO4", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Ammonium Hydroxide", "NH4OH", "Soluble"));
        compounds.Add(new CompoundSolubility("Magnesium Chloride", "MgCl2", "Soluble"));
        compounds.Add(new CompoundSolubility("Iron(II) Sulfate", "FeSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Barium Hydroxide", "Ba(OH)2", "Soluble"));
        compounds.Add(new CompoundSolubility("Iron(III) Sulfide", "Fe2S3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Iodide", "PbI2", "Insoluble"));
        compounds.Add(new CompoundSolubility("Sodium Sulfate", "Na2SO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Chloride", "KCl", "Soluble"));
        compounds.Add(new CompoundSolubility("Sodium Acetate", "CH3COONa", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Phosphate", "Ca3(PO4)2", "Insoluble"));
        compounds.Add(new CompoundSolubility("Aluminum Hydroxide", "Al(OH)3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Magnesium Sulfate", "MgSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Bromide", "KBr", "Soluble"));
        compounds.Add(new CompoundSolubility("Silver Iodide", "AgI", "Insoluble"));
        compounds.Add(new CompoundSolubility("Barium Nitrate", "Ba(NO3)2", "Soluble"));
        compounds.Add(new CompoundSolubility("Ammonium Chloride", "NH4Cl", "Soluble"));
        compounds.Add(new CompoundSolubility("Zinc Chloride", "ZnCl2", "Soluble"));
        compounds.Add(new CompoundSolubility("Mercury(I) Chloride", "Hg2Cl2", "Insoluble"));
        compounds.Add(new CompoundSolubility("Mercury(II) Sulfate", "HgSO4", "Insoluble"));
        compounds.Add(new CompoundSolubility("Copper(I) Iodide", "CuI", "Insoluble"));
        compounds.Add(new CompoundSolubility("Strontium Hydroxide", "Sr(OH)2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Magnesium Carbonate", "MgCO3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Sulfate", "PbSO4", "Insoluble"));
        compounds.Add(new CompoundSolubility("Calcium Nitrate", "Ca(NO3)2", "Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Iodide", "KI", "Soluble"));
        compounds.Add(new CompoundSolubility("Sodium Sulfide", "Na2S", "Soluble"));
        compounds.Add(new CompoundSolubility("Iron(III) Chloride", "FeCl3", "Soluble"));
        compounds.Add(new CompoundSolubility("Copper(II) Chloride", "CuCl2", "Soluble"));
        compounds.Add(new CompoundSolubility("Magnesium Nitrate", "Mg(NO3)2", "Soluble"));
        compounds.Add(new CompoundSolubility("Zinc Phosphate", "Zn3(PO4)2", "Insoluble"));
        compounds.Add(new CompoundSolubility("Cobalt(II) Chloride", "CoCl2", "Soluble"));
        compounds.Add(new CompoundSolubility("Sodium Phosphate", "Na3PO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Sulfate", "K2SO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Calcium Iodide", "CaI2", "Soluble"));
        compounds.Add(new CompoundSolubility("Ammonium Carbonate", "(NH4)2CO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Lithium Carbonate", "Li2CO3", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Silver Sulfate", "Ag2SO4", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Bromide", "PbBr2", "Slightly Soluble"));
        compounds.Add(new CompoundSolubility("Beryllium Hydroxide", "Be(OH)2", "Insoluble"));
        compounds.Add(new CompoundSolubility("Potassium Carbonate", "K2CO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Copper(I) Chloride", "CuCl", "Insoluble"));
        compounds.Add(new CompoundSolubility("Zinc Iodide", "ZnI2", "Soluble"));
        compounds.Add(new CompoundSolubility("Nickel(II) Hydroxide", "Ni(OH)2", "Insoluble"));
        compounds.Add(new CompoundSolubility("Chromium(III) Hydroxide", "Cr(OH)3", "Insoluble"));
        compounds.Add(new CompoundSolubility("Nickel(II) Chloride", "NiCl2", "Soluble"));
        compounds.Add(new CompoundSolubility("Silver Sulfide", "Ag2S", "Insoluble"));
        compounds.Add(new CompoundSolubility("Magnesium Bromide", "MgBr2", "Soluble"));
        compounds.Add(new CompoundSolubility("Potassium Dichromate", "K2Cr2O7", "Soluble"));
        compounds.Add(new CompoundSolubility("Cobalt(II) Sulfate", "CoSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Aluminum Chloride", "AlCl3", "Soluble"));
        compounds.Add(new CompoundSolubility("Nickel(II) Sulfate", "NiSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Manganese(II) Sulfate", "MnSO4", "Soluble"));
        compounds.Add(new CompoundSolubility("Zinc Oxide", "ZnO", "Insoluble"));
        compounds.Add(new CompoundSolubility("Sodium Bicarbonate", "NaHCO3", "Soluble"));
        compounds.Add(new CompoundSolubility("Lead(II) Chromate", "PbCrO4", "Insoluble"));

        // Continue adding more compounds as necessary...
    }
}
