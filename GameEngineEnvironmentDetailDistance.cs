using UnityEngine;

public class GameEngineEnvironmentDetailDistance : MonoBehaviour
{
    [SerializeField]
    private string level = "Medium";  // Default environment detail distance level

    private DetailDistanceSettings detailDistance;

    // Class to represent environment detail distance settings
    private class DetailDistanceSettings
    {
        public int distance;
        public string description;

        public DetailDistanceSettings(int distance, string description)
        {
            this.distance = distance;
            this.description = description;
        }
    }

    void Start()
    {
        detailDistance = DetermineDetailDistance(level);
        DisplayInfo();  // Display initial settings
    }

    // Method to determine environment detail distance based on quality level
    private DetailDistanceSettings DetermineDetailDistance(string level)
    {
        switch (level)
        {
            case "Low":
                return new DetailDistanceSettings(500, "Short distance rendering, lower detail at distance.");
            case "High":
                return new DetailDistanceSettings(2000, "High detail at long distances, increased load.");
            case "Ultra":
                return new DetailDistanceSettings(4000, "Maximum detail at maximum distances, very high load.");
            case "Medium":
            default:
                return new DetailDistanceSettings(1000, "Balanced detail and performance.");
        }
    }

    // Property to get and set environment detail distance level
    public string EnvironmentDetailDistance
    {
        get { return level; }
        set
        {
            level = value;
            detailDistance = DetermineDetailDistance(level);
            Debug.Log($"Environment Detail Distance Level set to: {level}");
        }
    }

    // Adjust the environment detail distance based on system performance preference
    public void AdjustDetailDistance(string performancePreference)
    {
        switch (performancePreference.ToLower())
        {
            case "low":
                EnvironmentDetailDistance = "Low";
                break;
            case "medium":
                EnvironmentDetailDistance = "Medium";
                break;
            case "high":
                EnvironmentDetailDistance = "High";
                break;
            default:
                Debug.LogWarning("Unknown performance preference. Using default detail distance.");
                EnvironmentDetailDistance = "Medium";
                break;
        }
    }

    // Display the current environment detail distance settings
    public void DisplayInfo()
    {
        Debug.Log($"Environment Detail Distance Level: {level}");
        Debug.Log($"Rendering Distance: {detailDistance.distance} units");
        Debug.Log($"Description: {detailDistance.description}");
    }
}
