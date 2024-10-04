using UnityEngine;

public class CharacterStats : MonoBehaviour
{
    // Array to store intelligence stats
    private float[] intelligences = new float[6];
    private string[] intelligenceNames = new string[6]
    {
        "Logical", "Linguistic", "Spatial", "Musical", "Kinesthetic", "Interpersonal"
    };

    // Other possible stats can be added here (e.g., health, stamina)
    public float health = 100f;
    public float stamina = 100f;

    // Initialization
    void Start()
    {
        // Initialize all intelligences with equal distribution (e.g., 16.66%)
        for (int i = 0; i < intelligences.Length; i++)
        {
            intelligences[i] = 100.0f / intelligences.Length;
        }

        // Call the function to display stats in the console
        DisplayCharacterStats();
    }

    // Function to adjust a specific intelligence
    public void AdjustIntelligence(int index, float adjustment)
    {
        // Ensure the adjustment keeps the value within the 0-100 range
        float newValue = Mathf.Clamp(intelligences[index] + adjustment, 0, 100);
        float delta = intelligences[index] - newValue;
        intelligences[index] = newValue;

        // Redistribute the delta among the remaining intelligences
        RedistributeIntelligence(index, -delta);

        // Display the updated stats after adjustment
        DisplayCharacterStats();
    }

    // Function to redistribute intelligence proportionally to the other stats
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

    // Display function to show the current stats in the console
    public void DisplayCharacterStats()
    {
        Debug.Log("==== Character Stats ====");
        Debug.Log("Health: " + health);
        Debug.Log("Stamina: " + stamina);
        
        // Display the intelligence stats
        for (int i = 0; i < intelligences.Length; i++)
        {
            Debug.Log(intelligenceNames[i] + " Intelligence: " + intelligences[i].ToString("F2") + "%");
        }
        Debug.Log("========================");
    }

    // Example method to reduce health (for future combat systems)
    public void TakeDamage(float damage)
    {
        health -= damage;
        health = Mathf.Clamp(health, 0, 100);
        Debug.Log("Took damage! Health is now: " + health);
    }

    // Example method to use stamina (for future movement or action systems)
    public void UseStamina(float amount)
    {
        stamina -= amount;
        stamina = Mathf.Clamp(stamina, 0, 100);
        Debug.Log("Used stamina! Stamina is now: " + stamina);
    }
}
