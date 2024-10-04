using UnityEngine;

public class CausalBodyParticleSystem : MonoBehaviour
{
    public Color auraColor = new Color(1f, 1f, 1f); // White for Causal Body
    public float auraTransparency = 0.3f; // Semi-transparent
    public float particleSize = 0.1f; // Size of individual particles
    public int particleCount = 2000; // Number of particles in the ring
    public float auraRadius = 4.5f; // Radius of the auric ring

    private ParticleSystem particleSystem;
    private Material particleMaterial;

    void Start()
    {
        // Create and configure the particle system for the aura
        SetupAuraParticleSystem();
    }

    void SetupAuraParticleSystem()
    {
        // Check if Particle System exists, if not, add one
        particleSystem = gameObject.GetComponent<ParticleSystem>();
        if (particleSystem == null)
        {
            particleSystem = gameObject.AddComponent<ParticleSystem>();
        }

        // Set particle system settings
        var main = particleSystem.main;
        main.startSize = particleSize; // Size of individual particles
        main.startColor = new Color(auraColor.r, auraColor.g, auraColor.b, auraTransparency);
        main.maxParticles = particleCount;
        main.loop = true;
        main.simulationSpace = ParticleSystemSimulationSpace.Local; // Keep particles relative to the object
        main.startSpeed = 0f; // No particle movement, particles stay in place

        // Set emission rate
        var emission = particleSystem.emission;
        emission.rateOverTime = particleCount;

        // Set particle shape to a Sphere (Ring)
        var shape = particleSystem.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere; // Ring shape
        shape.radius = auraRadius; // Radius of the aura ring
        shape.arcMode = ParticleSystemShapeMultiModeValue.Random; // Randomize particle placement along the ring

        // Disable any velocity or external force
        var velocityOverLifetime = particleSystem.velocityOverLifetime;
        velocityOverLifetime.enabled = false;

        var forceOverLifetime = particleSystem.forceOverLifetime;
        forceOverLifetime.enabled = false;

        // Create and apply a material to the particle system
        particleMaterial = new Material(Shader.Find("Particles/Standard Unlit"));
        particleMaterial.color = new Color(auraColor.r, auraColor.g, auraColor.b, auraTransparency);

        // Assign the material to the particle system
        var renderer = particleSystem.GetComponent<ParticleSystemRenderer>();
        renderer.material = particleMaterial;
    }
}
