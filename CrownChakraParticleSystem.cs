using UnityEngine;

public class CrownChakraParticleSystem : MonoBehaviour
{
    public Color chakraColor = new Color(0.8f, 0.4f, 1f); // Violet for Crown Chakra
    public float chakraTransparency = 1.0f; // Fully opaque
    public float particleSize = 0.05f; // Size of individual particles
    public int particleCount = 1000; // Number of particles in the system

    private ParticleSystem particleSystem;
    private Material particleMaterial;

    void Start()
    {
        // Create and configure the particle system for the chakra
        SetupChakraParticleSystem();
    }

    void SetupChakraParticleSystem()
    {
        // Check if Particle System exists, if not, add one
        particleSystem = gameObject.GetComponent<ParticleSystem>();
        if (particleSystem == null)
        {
            particleSystem = gameObject.AddComponent<ParticleSystem>();
        }

        // Set particle system settings
        var main = particleSystem.main;
        main.startSize = particleSize; // Small particles
        main.startColor = new Color(chakraColor.r, chakraColor.g, chakraColor.b, chakraTransparency);
        main.maxParticles = particleCount;
        main.loop = true;
        main.startLifetime = Mathf.Infinity; // Particles stay in place
        main.simulationSpace = ParticleSystemSimulationSpace.Local; // Make sure the particles stay local
        main.startSpeed = 0f; // No movement, particles stay in place

        // Set emission rate
        var emission = particleSystem.emission;
        emission.rateOverTime = particleCount;

        // Set particle shape to a small sphere to create a dense ball
        var shape = particleSystem.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = 0.2f; // Small radius to keep particles close together

        // Disable any velocity or external force
        var velocityOverLifetime = particleSystem.velocityOverLifetime;
        velocityOverLifetime.enabled = false;

        var forceOverLifetime = particleSystem.forceOverLifetime;
        forceOverLifetime.enabled = false;

        // Create and apply a material to the particle system
        particleMaterial = new Material(Shader.Find("Particles/Standard Unlit"));
        particleMaterial.color = new Color(chakraColor.r, chakraColor.g, chakraColor.b, chakraTransparency);

        // Assign the material to the particle system
        var renderer = particleSystem.GetComponent<ParticleSystemRenderer>();
        renderer.material = particleMaterial;
    }
}
