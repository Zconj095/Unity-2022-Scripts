using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class EnsembleDynamics : MonoBehaviour {
    public int numSpins = 5;
    public List<float> couplings = new List<float> { 2f, -1f, 3f, -0.5f };

    public List<float> ComputeSpinDynamics() {
        return Enumerable.Range(0, numSpins).Select(i => Mathf.Cos(Time.time + i)).ToList();
    }

    void Update() {
        List<float> spinDynamics = ComputeSpinDynamics();
        Debug.Log(string.Join(", ", spinDynamics));
    }
}