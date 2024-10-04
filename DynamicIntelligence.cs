using UnityEngine;

public class DynamicIntelligence : MonoBehaviour
{
    // Array to hold the percentage of each intelligence
    private float[] intelligences = new float[6];
    private string[] intelligenceNames = new string[6]
    {
        "Logical", "Linguistic", "Spatial", "Musical", "Kinesthetic", "Interpersonal"
    };
    
    // Start function initializes equal distribution (each intelligence is 16.66%)
    void Start()
    {
        for (int i = 0; i < intelligences.Length; i++)
        {
            intelligences[i] = 100.0f / intelligences.Length;
        }
        DisplayIntelligences();
    }
    
    // Function to adjust a particular intelligence by index
    public void AdjustIntelligence(int index, float adjustment)
    {
        // Ensure the adjustment keeps the value within the 0-100 range
        float totalOther = 0;
        for (int i = 0; i < intelligences.Length; i++)
        {
            if (i != index) totalOther += intelligences[i];
        }

        // Apply adjustment and balance the rest
        float newValue = Mathf.Clamp(intelligences[index] + adjustment, 0, 100);
        float delta = intelligences[index] - newValue;
        intelligences[index] = newValue;

        // Redistribute the delta to other intelligences proportionally
        RedistributeIntelligence(index, -delta);
        DisplayIntelligences();
    }

    // Function to redistribute the difference proportionally among the other intelligences
    private void RedistributeIntelligence(int excludeIndex, float delta)
    {
        float totalOther = 0;
        for (int i = 0; i < intelligences.Length; i++)
        {
            if (i != excludeIndex) totalOther += intelligences[i];
        }

        for (int i = 0; i < intelligences.Length; i++)
        {
            if (i != excludeIndex)
            {
                float proportionalChange = (intelligences[i] / totalOther) * delta;
                intelligences[i] = Mathf.Clamp(intelligences[i] + proportionalChange, 0, 100);
            }
        }
    }

    // Helper function to display the current intelligence distribution in the console
    private void DisplayIntelligences()
    {
        Debug.Log("Current Intelligence Distribution:");
        for (int i = 0; i < intelligences.Length; i++)
        {
            Debug.Log(intelligenceNames[i] + ": " + intelligences[i].ToString("F2") + "%");
        }
    }
}
