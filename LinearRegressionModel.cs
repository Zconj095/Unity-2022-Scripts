using UnityEngine;
using System.Linq;

public class LinearRegressionModel : MonoBehaviour {
    public void Fit(float[][] X, float[] y) {
        int m = y.Length;
        float[] xBar = new float[X[0].Length];
        float yBar = y.Average();
        float[] B = new float[X[0].Length]; // for slopes

        for (int i = 0; i < xBar.Length; i++) {
            xBar[i] = X.Average(col => col[i]);
        }

        for (int j = 0; j < xBar.Length; j++) {
            float numerator = 0;
            float denominator = 0;
            for (int i = 0; i < m; i++) {
                numerator += (X[i][j] - xBar[j]) * (y[i] - yBar);
                denominator += (X[i][j] - xBar[j]) * (X[i][j] - xBar[j]);
            }
            B[j] = numerator / denominator;
        }

        // Intercept
        float intercept = yBar;
        foreach (var coef in B) {
            intercept -= coef * xBar[AverageIndexOf(coef, B)];
        }

        Debug.Log("Regression Fitted with Intercept: " + intercept);
    }

    int AverageIndexOf(float value, float[] array) => array.ToList().IndexOf(value);
}