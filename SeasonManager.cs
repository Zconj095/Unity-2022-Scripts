using UnityEngine;
using System.Collections;
public class SeasonManager : MonoBehaviour
{
    public enum Season { Spring, Summer, Autumn, Winter }
    public Season currentSeason;
    public float seasonLength = 300f; // Length of each season in seconds

    private float seasonTimer;

    void Update()
    {
        seasonTimer += Time.deltaTime;
        if (seasonTimer >= seasonLength)
        {
            seasonTimer = 0f;
            ChangeSeason();
        }
    }

    void ChangeSeason()
    {
        currentSeason = (Season)(((int)currentSeason + 1) % 4);
        Debug.Log("Season changed to: " + currentSeason);

        // Add logic here to change environment based on season
    }
}
