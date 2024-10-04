using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class ContinuityEncoder : Agent
{
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Implementation of reaction to actions, usually moving or turning the agent
        // For now, we will assume that actions[0] is move forward intensity and actions[1] is turn intensity
        float moveIntensity = actions.ContinuousActions[0]; // Expected range [-1, 1]
        float turnIntensity = actions.ContinuousActions[1]; // Expected range [-1, 1]
        transform.Translate(Vector3.forward * moveIntensity * Time.deltaTime);
        transform.Rotate(Vector3.up, turnIntensity * 30 * Time.deltaTime); // Turn by up to 30 degrees per second
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetKey(KeyCode.W) ? 1.0f : 0.0f; // Move forward when 'W' is held
        continuousActionsOut[1] = Input.GetKey(KeyCode.D) ? 1.0f : (Input.GetKey(KeyCode.A) ? -1.0f : 0.0f); // Turn right with 'D', left with 'A'
    }
}