using UnityEngine;
using System.Linq;

public class ResonanceModeling : MonoBehaviour {
    void Start() {
        float[] t = Enumerable.Range(0, 1000).Select(i => i * 20f / 999f).ToArray();
        var results = ResonantInteractions(t, new float[]{0, Mathf.PI / 2});
        Debug.Log($"F1: {results.Item1[0]}, F2: {results.Item2[0]}, Interaction: {results.Item3[0]}");
    }

    (float[], float[], float[]) ResonantInteractions(float[] t, float[] phase) {
        float[] f1 = t.Select(x => 3 * Mathf.Sin(1 * (x + phase[0]))).ToArray();
        float[] f2 = t.Select(x => Mathf.Sin(3 * (x + phase[1]))).ToArray();
        float[] interaction = f1.Zip(f2, (x, y) => x * y).ToArray();

        return (f1, f2, interaction);
    }
}