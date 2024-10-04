using UnityEngine;

public class ColorAdjustmentManager : MonoBehaviour
{
    public Material adjustmentMaterial;
    private bool isGameRunning = true;

    void Start()
    {
        // Initialize default values or adjust them based on user data
        SetColorAdjustments(1.0f, 1.0f, 1.0f);
    }

    void Update()
    {
        if (isGameRunning)
        {
            // Assume methods to get these values exist
            string colorblindnessType = DetermineColorblindnessType();
            ApplyColorAdjustments(colorblindnessType);
        }
    }

    private string DetermineColorblindnessType()
    {
        // This should be replaced with actual data gathering and analysis
        return "normal"; // Placeholder
    }

    private void ApplyColorAdjustments(string colorblindnessType)
    {
        switch (colorblindnessType)
        {
            case "protanopia":
                SetColorAdjustments(0.5f, 1.0f, 1.0f);
                break;
            case "deuteranopia":
                SetColorAdjustments(1.0f, 0.5f, 1.0f);
                break;
            case "tritanopia":
                SetColorAdjustments(1.0f, 1.0f, 0.5f);
                break;
            default:
                SetColorAdjustments(1.0f, 1.0f, 1.0f); // Normal vision
                break;
        }
    }

    private void SetColorAdjustments(float redShift, float greenShift, float blueShift)
    {
        adjustmentMaterial.SetFloat("_RedShift", redShift);
        adjustmentMaterial.SetFloat("_GreenShift", greenShift);
        adjustmentMaterial.SetFloat("_BlueShift", blueShift);
    }
}