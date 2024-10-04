using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// Equivalent to OscillationAnalyzer
public class OscillationAnalyzer: MonoBehaviour
{
    private float[] timestamps;
    private float[] values;

    public OscillationAnalyzer(float[] ts, float[] val)
    {
        timestamps = ts;
        values = val;
    }

    public float[] LombScargle(float[] freqs)
    {
        int n = timestamps.Length;
        float[] P = new float[freqs.Length];
        float[] sin2wt, cos2wt, sinwt, coswt, tau;
        
        sin2wt = new float[n];
        cos2wt = new float[n];
        sinwt = new float[n];
        coswt = new float[n];
        tau = new float[freqs.Length];

        for (int j = 0; j < freqs.Length; j++)
        {
            float omega = 2 * Mathf.PI * freqs[j];

            float sumSin = 0, sumCos = 0;

            for (int i = 0; i < n; i++)
            {
                sinwt[i] = Mathf.Sin(omega * timestamps[i]);
                coswt[i] = Mathf.Cos(omega * timestamps[i]);
                sumSin += sinwt[i];
                sumCos += coswt[i];
            }

            float sumSin2 = sumSin * sumSin / n;
            float sumCos2 = sumCos * sumCos / n;

            tau[j] = Mathf.Atan2(2 * sumSin, 2 * sumCos) / (2 * omega);
            
            float sinTau, cosTau;
            float sumYC = 0, sumYS = 0, sumCC = 0, sumSS = 0, sumCS = 0;
            
            for (int i = 0; i < n; i++)
            {
                sinTau = Mathf.Sin(omega * timestamps[i] - tau[j]);
                cosTau = Mathf.Cos(omega * timestamps[i] - tau[j]);
                sumYC += values[i] * cosTau;
                sumYS += values[i] * sinTau;
                sumCC += cosTau * cosTau;
                sumSS += sinTau * sinTau;
                sumCS += cosTau * sinTau;
            }
            
            float D = sumCC * sumSS - sumCS * sumCS;
            float C = (sumYC * sumSS - sumYS * sumCS) / D;
            float S = (sumYS * sumCC - sumYC * sumCS) / D;
            
            P[j] = 0.5f * ((sumYC * C + sumYS * S) / n);
        }
        return P;
    }
}