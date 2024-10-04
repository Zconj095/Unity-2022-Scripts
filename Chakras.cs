using UnityEngine;

public class Chakras : MonoBehaviour
{
    // Define properties for each chakra's energy level
    public float rootChakraEnergy = 0.5f;
    public float sacralChakraEnergy = 0.5f;
    public float solarPlexusChakraEnergy = 0.5f;
    public float heartChakraEnergy = 0.5f;
    public float throatChakraEnergy = 0.5f;
    public float thirdEyeChakraEnergy = 0.5f;
    public float crownChakraEnergy = 0.5f;

    // Method to increase energy level of a specific chakra
    public void IncreaseChakraEnergy(string chakraName, float amount)
    {
        switch (chakraName.ToLower())
        {
            case "root":
                rootChakraEnergy = Mathf.Clamp01(rootChakraEnergy + amount);
                break;
            case "sacral":
                sacralChakraEnergy = Mathf.Clamp01(sacralChakraEnergy + amount);
                break;
            case "solarplexus":
                solarPlexusChakraEnergy = Mathf.Clamp01(solarPlexusChakraEnergy + amount);
                break;
            case "heart":
                heartChakraEnergy = Mathf.Clamp01(heartChakraEnergy + amount);
                break;
            case "throat":
                throatChakraEnergy = Mathf.Clamp01(throatChakraEnergy + amount);
                break;
            case "thirdeye":
                thirdEyeChakraEnergy = Mathf.Clamp01(thirdEyeChakraEnergy + amount);
                break;
            case "crown":
                crownChakraEnergy = Mathf.Clamp01(crownChakraEnergy + amount);
                break;
            default:
                Debug.LogWarning("Invalid chakra name.");
                break;
        }
    }

    // Method to decrease energy level of a specific chakra
    public void DecreaseChakraEnergy(string chakraName, float amount)
    {
        IncreaseChakraEnergy(chakraName, -amount);
    }
}
