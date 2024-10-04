using UnityEngine;

public class TimeOfDayManager : MonoBehaviour
{
    public Light sun;
    public float dayLength = 1200f; // Length of day in seconds
    private float timeOfDay;

    void Update()
    {
        timeOfDay += Time.deltaTime / dayLength;
        if (timeOfDay > 1f)
            timeOfDay = 0f;

        UpdateSunPosition();
    }

    void UpdateSunPosition()
    {
        float angle = timeOfDay * 360f;
        sun.transform.rotation = Quaternion.Euler(new Vector3(angle - 90, 170, 0));
        RenderSettings.ambientIntensity = Mathf.Clamp01(1.0f - Mathf.Abs(timeOfDay - 0.5f) * 2f);
    }
}
