using UnityEngine;
[System.Serializable]
public class ElectromagneticParticle: MonoBehaviour
{
    public string elementName;
    public float mass;  // Mass in atomic mass units (amu)
    public float charge;  // Charge in elementary charges (e.g., 1 for protons, -1 for electrons)
    public Color particleColor;  // Color for visualization

    public float magneticMoment;  // Magnetic moment in some units

    public ElectromagneticParticle(string elementName, float mass, Color particleColor, float charge, float magneticMoment)
    {
        this.elementName = elementName;
        this.mass = mass;
        this.particleColor = particleColor;
        this.charge = charge;
        this.magneticMoment = magneticMoment;
    }

    // Method to calculate Lorentz force F = q(E + v × B)
    public Vector3 CalculateLorentzForce(Vector3 electricField, Vector3 velocity, Vector3 magneticField)
    {
        Vector3 electricForce = charge * electricField;
        Vector3 magneticForce = charge * Vector3.Cross(velocity, magneticField);
        return electricForce + magneticForce;  // Total force
    }

}
