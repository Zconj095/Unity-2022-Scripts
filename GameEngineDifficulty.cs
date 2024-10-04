using UnityEngine;
using System.Collections.Generic;

public class GameEngineDifficulty : MonoBehaviour
{
    [SerializeField]
    private string level = "Normal";  // Default difficulty level

    private Dictionary<string, float> difficultyModifiers;

    void Start()
    {
        difficultyModifiers = SetDifficultyModifiers(level);
        DisplayInfo();  // Display initial settings
    }

    // Method to set difficulty modifiers based on the chosen level
    private Dictionary<string, float> SetDifficultyModifiers(string level)
    {
        var modifiers = new Dictionary<string, Dictionary<string, float>>()
        {
            { "Easy", new Dictionary<string, float> { {"enemy_strength", 0.75f}, {"resource_abundance", 1.5f}, {"puzzle_complexity", 0.75f} } },
            { "Normal", new Dictionary<string, float> { {"enemy_strength", 1.0f}, {"resource_abundance", 1.0f}, {"puzzle_complexity", 1.0f} } },
            { "Hard", new Dictionary<string, float> { {"enemy_strength", 1.5f}, {"resource_abundance", 0.75f}, {"puzzle_complexity", 1.25f} } },
            { "Expert", new Dictionary<string, float> { {"enemy_strength", 2.0f}, {"resource_abundance", 0.5f}, {"puzzle_complexity", 1.5f} } }
        };
        return modifiers.ContainsKey(level) ? modifiers[level] : modifiers["Normal"];
    }

    // Property to get and set difficulty level
    public string DifficultyLevel
    {
        get { return level; }
        set
        {
            if (value != "Easy" && value != "Normal" && value != "Hard" && value != "Expert")
            {
                Debug.LogError("Invalid difficulty level. Choose 'Easy', 'Normal', 'Hard', or 'Expert'.");
            }
            else
            {
                level = value;
                difficultyModifiers = SetDifficultyModifiers(level);
                Debug.Log($"Difficulty set to: {level}");
            }
        }
    }

    // Adjust the difficulty based on player preference
    public void AdjustDifficulty(string playerPreference)
    {
        switch (playerPreference.ToLower())
        {
            case "easier":
                DifficultyLevel = "Easy";
                break;
            case "normal":
                DifficultyLevel = "Normal";
                break;
            case "harder":
                DifficultyLevel = "Hard";
                break;
            default:
                Debug.LogWarning("Unknown preference. Using default difficulty.");
                DifficultyLevel = "Normal";
                break;
        }
    }

    // Display the current difficulty level and its effects on gameplay
    public void DisplayInfo()
    {
        Debug.Log($"Difficulty Level: {level}");
        Debug.Log("Modifiers:");
        foreach (var modifier in difficultyModifiers)
        {
            Debug.Log($"  {modifier.Key}: {modifier.Value}");
        }
    }
}
