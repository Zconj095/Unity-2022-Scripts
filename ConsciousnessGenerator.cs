using UnityEngine;

public class ConsciousnessGenerator : MonoBehaviour
{
    [Header("Consciousness States")]
    public GameObject[] consciousnessStateObjects; // Array to store generated consciousness state GameObjects
    public string[] consciousnessStates = { "Conscious", "Unconscious", "Non-conscious", "Subconscious", "Superconscious" };

    void Start()
    {
        consciousnessStateObjects = new GameObject[consciousnessStates.Length]; // Initialize array
    }

    [ContextMenu("Generate Consciousness States")]
    public void GenerateConsciousnessStates()
    {
        for (int i = 0; i < consciousnessStates.Length; i++)
        {
            CreateConsciousnessState(consciousnessStates[i], i);
        }
    }

    void CreateConsciousnessState(string stateName, int index)
    {
        GameObject state = new GameObject(stateName);
        state.transform.position = new Vector3(index * 2.0f, 0, 0);
        consciousnessStateObjects[index] = state; // Store reference to the generated state
    }
}
