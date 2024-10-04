using UnityEngine;

[System.Serializable]
public class ElectrodePotential: MonoBehaviour
{
    public string halfReaction;  // The half-reaction (e.g., "Cu²⁺ + 2e⁻ → Cu")
    public float standardPotential;  // The standard reduction potential in volts (V)

    public ElectrodePotential(string halfReaction, float standardPotential)
    {
        this.halfReaction = halfReaction;
        this.standardPotential = standardPotential;
    }
}
