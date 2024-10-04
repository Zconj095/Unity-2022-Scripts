using UnityEngine;
using System.Collections.Generic;

public class AuraSource : MonoBehaviour {

    public Dictionary<string, float> signal = new Dictionary<string, float>();
    private float[] rawSignal;

    void Start() {
        signal.Add("freq", Random.Range(1, 12));
        signal.Add("phase", Random.value);
        signal.Add("amplitude", Random.value);
        rawSignal = GenerateSignal();
    }

    private float[] GenerateSignal() {
        float[] rawData = new float[100];
        for (int i = 0; i < 100; i++) {
            rawData[i] = signal["amplitude"] * Mathf.Sin((signal["freq"] / 1.3f) * signal["phase"] + Random.value * Mathf.PI);
        }
        return rawData;
    }
}

