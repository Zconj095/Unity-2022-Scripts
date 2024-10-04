using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class MultivariateRegression : MonoBehaviour {
    
    public void Start() {
        float[,] X = new float[,] { { 1, 2 }, { 3, 1 } }; // Example Data
        float[] Y = new float[] { 3, 5 };                 // Target

        var reg = new SimpleLinearRegression();
        reg.Fit(X, Y);
    }

    public class SimpleLinearRegression {
        // Dummy simple linear regression for explanation (doesn't actually compute regression)
        public void Fit(float[,] X, float[] y) {
            Debug.Log("Fit model here");
        }
    }
}