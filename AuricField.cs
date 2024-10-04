using UnityEngine;
using System.Collections.Generic;

public class AuricField : MonoBehaviour {
    Dictionary<string, float> auricFrequencies = new Dictionary<string, float> {
        {"Physical", 0.01f},
        {"Emotional", 0.1f},
        {"Mental", 0.4f},
        {"Spiritual", 2f}
    };

    Dictionary<string, string> endocrineMapping = new Dictionary<string, string> {
        {"Root", "Adrenal"}, {"Sacral", "Gonads"}, {"Solar Plexus", "Pancreas"},
        {"Heart", "Thymus"}, {"Throat", "Thyroid"}, {"Third Eye", "Pituitary"}, {"Crown", "Pineal"}
    };
    
    Dictionary<string, float> chakraFrequencies = new Dictionary<string, float> {
        {"Root", 0.1f}, {"Sacral", 0.2f}, {"Solar Plexus", 0.3f},
        {"Heart", 0.5f}, {"Throat", 0.7f}, {"Third Eye", 0.9f}, {"Crown", 1f}
    };

    void Start() {
        Debug.Log("Auric Frequencies and Chakras Initialized.");
    }
}