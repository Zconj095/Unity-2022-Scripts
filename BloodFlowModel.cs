using UnityEngine;

public class BloodFlowModel : MonoBehaviour
{
    float r = 0.1f;  // vascular resistance
    float l = 0.5f;  // vessel length
    float qIn = 5f;  // mL/s

    float FlowRate(float pIn, float pOut)
    {
        return (pIn - pOut) / (r * l);
    }

    float PressureDrop(float flow)
    {
        return flow * r * l;
    }

    public float[] Simulate(int timesteps)
    {
        float[] pressures = new float[timesteps];
        float dp = 0;
        float p1 = 100; // start pressure
        float p2 = 0;

        for (int t = 0; t < timesteps; t++)
        {
            float dq1_dt = FlowRate(p1, p2);
            dp = PressureDrop(dq1_dt);
            p1 -= dp;
            p2 += dp;
            pressures[t] = p1;
        }

        return pressures;
    }
}