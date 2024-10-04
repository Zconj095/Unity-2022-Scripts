using UnityEngine;

public class IdleBehaviorManager : MonoBehaviour
{
    public float idleTimeThreshold = 10f; // Time in seconds before idle actions start
    private float idleTimer;

    void Update()
    {
        if (Input.anyKeyDown)
        {
            idleTimer = 0f;
            return;
        }

        idleTimer += Time.deltaTime;

        if (idleTimer >= idleTimeThreshold)
        {
            TriggerIdleBehavior();
            idleTimer = 0f;
        }
    }

    void TriggerIdleBehavior()
    {
        Debug.Log("Player is idle, triggering idle behavior.");
        // Insert animation or action here, e.g., character looks around or stretches
    }
}
