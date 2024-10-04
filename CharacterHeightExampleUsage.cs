using System.Collections.Generic;
using UnityEngine;

public class CharacterHeightExampleUsage : MonoBehaviour
{
    private CharacterHeight characterHeight;

    void Start()
    {
        // Initialize the CharacterHeight component
        characterHeight = gameObject.AddComponent<CharacterHeight>();
        
        // Display initial height
        characterHeight.DisplayHeight();

        // Set and display manual height
        characterHeight.SetManualHeight(1.85f);
        characterHeight.DisplayHeight();

        // Set and display scaled height
        float scaledHeight = characterHeight.SetScaledHeight(1.2f);
        Debug.Log($"Scaled Height: {scaledHeight} meters");

        // Display final height
        characterHeight.DisplayHeight();
    }
}
