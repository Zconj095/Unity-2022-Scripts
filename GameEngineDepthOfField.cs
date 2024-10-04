using UnityEngine;

public class GameEngineDepthOfField : MonoBehaviour
{
    [SerializeField]
    private bool enabled = true;  // Default DOF enabled state

    [SerializeField]
    private float focalDistance = 10.0f;  // Default focal distance in units

    [SerializeField]
    private float blurIntensity = 1.0f;  // Default blur intensity

    // Property to get and set DOF settings
    public (bool enabled, float focalDistance, float blurIntensity) DOFSettings
    {
        get { return (enabled, focalDistance, blurIntensity); }
        set
        {
            enabled = value.enabled;
            focalDistance = value.focalDistance;
            blurIntensity = value.blurIntensity;
            Debug.Log($"DOF Settings updated: Enabled = {enabled}, Focal Distance = {focalDistance} units, Blur Intensity = {blurIntensity}");
        }
    }

    // Adjust the DOF settings based on whether cinematic mode is enabled
    public void AdjustDepthOfField(bool cinematicMode = false)
    {
        if (cinematicMode)
        {
            focalDistance = 5.0f;
            blurIntensity = 2.0f;
        }
        else
        {
            focalDistance = 10.0f;
            blurIntensity = 1.0f;
        }
        Debug.Log($"DOF adjusted for cinematic mode: Focal Distance = {focalDistance} units, Blur Intensity = {blurIntensity}");
    }

    // Display the current DOF settings
    public void DisplayInfo()
    {
        string status = enabled ? "enabled" : "disabled";
        Debug.Log($"Depth of Field: {status}");
        Debug.Log($"Focal Distance: {focalDistance} units");
        Debug.Log($"Blur Intensity: {blurIntensity}");
    }

    void Start()
    {
        DisplayInfo();  // Display initial settings
    }
}
