using System.Collections.Generic;
using UnityEngine;
[System.Serializable]
public class CompoundSolubility: MonoBehaviour
{
    public string compoundName;  // Name of the compound
    public string formula;       // Chemical formula
    public string solubility;    // Solubility status (Soluble, Insoluble, Slightly Soluble)

    public CompoundSolubility(string compoundName, string formula, string solubility)
    {
        this.compoundName = compoundName;
        this.formula = formula;
        this.solubility = solubility;
    }
}
