using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine;

public class CognitiveFunctionTester : MonoBehaviour
{
    void Start()
    {
        // Example: Get the cognitive functions for an INFJ
        PersonalityType type = PersonalityType.INFJ;
        CognitiveFunction[] functions = MyersBriggsFunctions.GetFunctionsForType(type);

        Debug.Log("Personality Type: " + type);
        foreach (CognitiveFunction function in functions)
        {
            Debug.Log("Cognitive Function: " + function);
        }
    }
}
