using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PersonalityDisplay : MonoBehaviour
{
    private MBTITypes mbtiTypes;

    // Start is called before the first frame update
    void Start()
    {
        // Get the MBTITypes component from the PersonalityManager GameObject
        mbtiTypes = GameObject.Find("PersonalityManager").GetComponent<MBTITypes>();

        // Example: Retrieve a personality by its name
        MBTITypes.PersonalityType infjType = mbtiTypes.GetPersonalityTypeByName("INFJ");
        Debug.Log("Personality Name: " + infjType.name + " - Code: " + infjType.code);

        // Example: Retrieve a personality by its code
        MBTITypes.PersonalityType architectType = mbtiTypes.GetPersonalityTypeByCode("The Architect");
        Debug.Log("Personality Code: " + architectType.code + " - Name: " + architectType.name);
    }
}
