using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class SensorArray : MonoBehaviour {
    public List<float[]> eeg = new List<float[]>();
    public List<float[]> ekg = new List<float[]>();

    public void Measure() {
        eeg.Add(new float[20].Select(_ => Random.Range(0f, 1f)).ToArray());
        ekg.Add(new float[5].Select(_ => Random.Range(0f, 1f) + 10f).ToArray());
    }

    public float Bpm {
        get {
            float[] lastEkg = ekg.LastOrDefault() ?? new float[0];
            int count = lastEkg.Count(f => f > 10.5f); // Count peaks as a simple example
            return count / 4f;
        }
    }
}