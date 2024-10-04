using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
public class HebbianNet : MonoBehaviour {
    public float[] weights;

    public void Train(float[] inputs) {
        for (int i = 0; i < inputs.Length; i++) {
            weights[i] += inputs[i]; // Simple Hebbian update
        }
    }
}
