using UnityEngine;

public class NeurotransmitterModel : MonoBehaviour
{
    public Vector3 Simulate(float[] timesteps, Vector3 initialConditions)
    {
        Vector3 currentCondition = initialConditions;
        foreach (float t in timesteps)
        {
            float GLU = currentCondition.x;
            float DA = currentCondition.y;
            float SE = currentCondition.z;

            float dg_dt = (-2) * GLU + SE;
            float dd_dt = (-6) * DA * DA + 3;
            float dse_dt = (-0.5f) * SE + (GLU / 2);

            // Update with small timestep (Euler method)
            currentCondition += new Vector3(dg_dt, dd_dt, dse_dt) * 0.01f; // Small delta time
        }
        return currentCondition;
    }
}