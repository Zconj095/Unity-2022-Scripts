using UnityEngine;

public class Ensemble : MonoBehaviour
{
    float coupling = 0.1f;

    public Vector2 EnsembleDynamics(Vector2 state, float time)
    {
        float x = state.x;
        float y = state.y;
        Vector2 derivs = new Vector2(-x + coupling * y, -y + coupling * x);
        return derivs;
    }
}