using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QuantumElementConnection: MonoBehaviour
{
    public QuantumElementThree targetElement;  // The element we're connected to
    public float transitionProbability;   // Probability of transitioning to this element

    // Constructor
    public QuantumElementConnection(QuantumElementThree target, float probability)
    {
        targetElement = target;
        transitionProbability = probability;
    }
}
