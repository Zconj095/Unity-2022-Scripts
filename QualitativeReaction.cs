using UnityEngine;

[System.Serializable]
public class QualitativeReaction: MonoBehaviour
{
    public string testName;        // Name of the qualitative test (e.g., "Flame Test")
    public string targetIon;       // Ion or compound being tested for (e.g., "Na⁺", "Cl⁻")
    public string reagent;         // Reagent used for the test (e.g., "Silver Nitrate for Chlorides")
    public string observableChange; // Observable outcome (e.g., "Yellow flame", "White precipitate")
    public string testResult;      // Result of the test (e.g., "Na⁺ confirmed", "Cl⁻ confirmed")

    public QualitativeReaction(string testName, string targetIon, string reagent, string observableChange, string testResult)
    {
        this.testName = testName;
        this.targetIon = targetIon;
        this.reagent = reagent;
        this.observableChange = observableChange;
        this.testResult = testResult;
    }
}
