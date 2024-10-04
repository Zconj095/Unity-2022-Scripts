using UnityEngine;
using System.Linq;

public class SDEMimic : MonoBehaviour {

    public float SimulateSDE(float initValue, float timesteps) {
        float state = initValue;
        float mu = 0.1f;
        float sigma = 1f;
        float dt = 10f / timesteps;

        for (int i = 0; i < timesteps; i++) {
            state += mu * state * dt + sigma * state * Random.Range(-1f, 1f) * Mathf.Sqrt(dt);
        }
        return state;
    }

    void Start() {
        float result = SimulateSDE(1.0f, 1000);
        Debug.Log($"Result of SDE: {result}");
    }
}