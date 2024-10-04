using UnityEngine;

public class SleepStateGenerator : MonoBehaviour
{
    [Header("Sleep States")]
    public GameObject[] sleepStateObjects; // Array to store generated sleep state GameObjects
    public string[] sleepStates = { "NREM 1", "NREM 2", "NREM 3", "NREM 4", "REM" };

    void Start()
    {
        sleepStateObjects = new GameObject[sleepStates.Length]; // Initialize array
    }

    [ContextMenu("Generate Sleep States")]
    public void GenerateSleepStates()
    {
        for (int i = 0; i < sleepStates.Length; i++)
        {
            CreateSleepState(sleepStates[i], i);
        }
    }

    void CreateSleepState(string stateName, int index)
    {
        GameObject state = new GameObject(stateName);
        state.transform.position = new Vector3(index * 2.0f, -2.0f, 0);
        sleepStateObjects[index] = state; // Store reference to the generated state
    }
}
