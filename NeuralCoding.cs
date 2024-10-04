using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine;

public class NeuralCoding: MonoBehaviour{
    // Encoding function F
    public static float[] F(float[] x) {
        float[] y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) {
            y[i] = 2 * x[i] + 3;
        }
        return y;
    }

    // Decoding function F^-1
    public static float[] F_inv(float[] y) {
        float[] x = new float[y.Length];
        for (int i = 0; i < y.Length; i++) {
            x[i] = (y[i] - 3) / 2;
        }
        return x;
    }

    // Narrow version of encoding
    public static float[] EncodingNarrow(float[] stimuli) {
        return F(stimuli);
    }

    // Wide version of encoding
    public static float[] EncodingWide(float[] stimuli, float[] bodyState, float[] brainActivity) {
        float[] neuralActivity = new float[stimuli.Length];
        for (int i = 0; i < stimuli.Length; i++) {
            neuralActivity[i] = 2 * stimuli[i] + bodyState[i] + 3 * brainActivity[i];
        }
        return neuralActivity;
    }

    // Prediction of neural activity error based on cellular properties and states
    public static float[] PredictError(float[] currentState, float[] previousState) {
        float[] error = new float[currentState.Length];
        for (int i = 0; i < currentState.Length; i++) {
            error[i] = currentState[i] - previousState[i] + GaussianNoise();
        }
        return error;
    }

    // Generate Gaussian-like noise using the Box-Muller transform
    private static float GaussianNoise() {
        float u1 = 1.0f - UnityEngine.Random.value; // uniform(0,1] random doubles
        float u2 = 1.0f - UnityEngine.Random.value;
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) *
                             Mathf.Sin(2.0f * Mathf.PI * u2); // random normal(0,1)
        return 0.1f * randStdNormal; // scale to 0.1 std dev
    }
}