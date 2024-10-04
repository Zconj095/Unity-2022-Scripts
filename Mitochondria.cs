using UnityEngine;
using System;

public class Mitochondria : MonoBehaviour
{
    public float membranePotential = -150.0f;

    public float ProduceATP()
    {
        float atp = 100.0f * Mathf.Exp(membranePotential / 1000.0f);
        return atp;
    }
}
