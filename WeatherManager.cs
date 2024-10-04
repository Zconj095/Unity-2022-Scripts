using UnityEngine;

public class WeatherManager : MonoBehaviour
{
    public enum WeatherType { Clear, Rain, Snow }
    public WeatherType currentWeather;
    public ParticleSystem rainEffect;
    public ParticleSystem snowEffect;

    void Update()
    {
        // Simple weather cycle for demonstration
        if (Input.GetKeyDown(KeyCode.R))
            SetWeather(WeatherType.Rain);
        if (Input.GetKeyDown(KeyCode.S))
            SetWeather(WeatherType.Snow);
        if (Input.GetKeyDown(KeyCode.C))
            SetWeather(WeatherType.Clear);
    }

    void SetWeather(WeatherType weather)
    {
        currentWeather = weather;
        rainEffect.Stop();
        snowEffect.Stop();

        switch (weather)
        {
            case WeatherType.Rain:
                rainEffect.Play();
                break;
            case WeatherType.Snow:
                snowEffect.Play();
                break;
        }
    }
}
