using System.Collections.Generic;
using UnityEngine;

public class ColorAdjustmentCalculator : MonoBehaviour
{
    public Dictionary<string, float> CalculateSpecificAdjustments(string identifiedType, Dictionary<string, float> specificAdjustmentNeeds)
    {
        var adjustments = new Dictionary<string, float>
        {
            {"red_shift", 1.0f},
            {"green_shift", 1.0f},
            {"blue_shift", 1.0f},
            {"contrast_increase", 1.0f},
            {"saturation_adjustment", 1.0f}
        };

        if (identifiedType == "protanopia")
        {
            adjustments["red_shift"] = specificAdjustmentNeeds.GetValueOrDefault("red_shift", 1.2f);
            adjustments["green_shift"] = specificAdjustmentNeeds.GetValueOrDefault("green_shift", 0.8f);
        }
        else if (identifiedType == "deuteranopia")
        {
            adjustments["green_shift"] = specificAdjustmentNeeds.GetValueOrDefault("green_shift", 1.2f);
            adjustments["red_shift"] = specificAdjustmentNeeds.GetValueOrDefault("red_shift", 0.8f);
        }
        else if (identifiedType == "tritanopia")
        {
            adjustments["blue_shift"] = specificAdjustmentNeeds.GetValueOrDefault("blue_shift", 1.2f);
        }
        else if (identifiedType == "monochrome")
        {
            adjustments["contrast_increase"] = specificAdjustmentNeeds.GetValueOrDefault("contrast_increase", 1.5f);
        }
        else if (identifiedType == "difficulty_purple")
        {
            adjustments["saturation_adjustment"] = specificAdjustmentNeeds.GetValueOrDefault("saturation_adjustment", 0.8f);
        }

        return adjustments;
    }
}