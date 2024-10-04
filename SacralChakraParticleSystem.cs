using UnityEngine;

public class SacralChakraParticleSystem : MonoBehaviour
{
    public Color chakraColor = new Color(1f, 0.6f, 0f); // Orange for Sacral Chakra
    public float chakraTransparency = 1.0f;
    public float particleSize = 0.05f;
    public int particleCount = 1000;

    private ParticleSystem particleSystem;
    private Material particleMaterial;

    void Start()
    {
        SetupChakraParticleSystem();
    }

    void SetupChakraParticleSystem()
    {
        particleSystem = gameObject.GetComponent<ParticleSystem>();
        if (particleSystem == null)
        {
            particleSystem = gameObject.AddComponent<ParticleSystem>();
        }

        var main = particleSystem.main;
        main.startSize = particleSize;
        main.startColor = new Color(chakraColor.r, chakraColor.g, chakraColor.b, chakraTransparency);
        main.maxParticles = particleCount;
        main.loop = true;
        main.startLifetime = Mathf.Infinity;
        main.startSpeed = 0f;

        var emission = particleSystem.emission;
        emission.rateOverTime = particleCount;

        var shape = particleSystem.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = 0.2f;

        var velocityOverLifetime = particleSystem.velocityOverLifetime;
        velocityOverLifetime.enabled = false;

        var forceOverLifetime = particleSystem.forceOverLifetime;
        forceOverLifetime.enabled = false;

        particleMaterial = new Material(Shader.Find("Particles/Standard Unlit"));
        particleMaterial.color = new Color(chakraColor.r, chakraColor.g, chakraColor.b, chakraTransparency);

        var renderer = particleSystem.GetComponent<ParticleSystemRenderer>();
        renderer.material = particleMaterial;
    }
}
