using UnityEngine;

[RequireComponent(typeof(EmissionSpectrumGenerator))]
public class EmissionSpectrumVisualizer : MonoBehaviour
{
    private EmissionSpectrumGenerator generator;

    void Start()
    {
        generator = GetComponent<EmissionSpectrumGenerator>();
    }

    void OnDrawGizmos()
    {
        if (generator != null && generator.emissionLines.Count > 0)
        {
            // Define the starting position for the spectrum bars
            Vector3 startPosition = transform.position;
            float barWidth = 0.2f; // Width of each emission line bar
            float barHeight = 2.0f; // Height of the emission line bar

            foreach (EmissionLine line in generator.emissionLines)
            {
                Gizmos.color = line.emissionColor;

                // Draw a colored bar for each emission line
                Gizmos.DrawCube(startPosition, new Vector3(barWidth, barHeight, 0.1f));

                // Move the starting position for the next bar
                startPosition += new Vector3(barWidth + 0.1f, 0, 0);
            }
        }
    }
}
