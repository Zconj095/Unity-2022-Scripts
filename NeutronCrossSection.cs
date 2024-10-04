using UnityEngine;

[System.Serializable]
public class NeutronCrossSection: MonoBehaviour
{
    public string isotopeName;        // Isotope name (e.g., "Uranium-235")
    public float absorptionCrossSection; // Absorption cross-section in barns
    public float scatteringCrossSection; // Scattering cross-section in barns
    public string reactionType;       // Type of reaction (e.g., "Absorption", "Scattering")

    public NeutronCrossSection(string isotopeName, float absorptionCrossSection, float scatteringCrossSection, string reactionType)
    {
        this.isotopeName = isotopeName;
        this.absorptionCrossSection = absorptionCrossSection;
        this.scatteringCrossSection = scatteringCrossSection;
        this.reactionType = reactionType;
    }
}
