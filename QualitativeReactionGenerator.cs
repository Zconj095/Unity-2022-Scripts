using System.Collections.Generic;
using UnityEngine;

public class QualitativeReactionGenerator : MonoBehaviour
{
    [Header("Qualitative Reactions")]
    public List<QualitativeReaction> qualitativeReactions = new List<QualitativeReaction>();

    void Start()
    {
        GenerateQualitativeReactions();
    }

    [ContextMenu("Generate Qualitative Reactions")]
    public void GenerateQualitativeReactions()
    {
        qualitativeReactions.Clear();  // Clear the list before generating

        // Add common qualitative reactions
        qualitativeReactions.Add(new QualitativeReaction("Flame Test", "Na⁺", "None", "Yellow flame", "Sodium ion (Na⁺) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Flame Test", "K⁺", "None", "Lilac flame", "Potassium ion (K⁺) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Flame Test", "Ca²⁺", "None", "Brick-red flame", "Calcium ion (Ca²⁺) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Silver Nitrate Test", "Cl⁻", "Silver Nitrate", "White precipitate", "Chloride ion (Cl⁻) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Silver Nitrate Test", "Br⁻", "Silver Nitrate", "Cream precipitate", "Bromide ion (Br⁻) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Silver Nitrate Test", "I⁻", "Silver Nitrate", "Yellow precipitate", "Iodide ion (I⁻) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Barium Chloride Test", "SO₄²⁻", "Barium Chloride", "White precipitate", "Sulfate ion (SO₄²⁻) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Limewater Test", "CO₃²⁻", "Hydrochloric Acid", "Bubbles of gas turning limewater cloudy", "Carbonate ion (CO₃²⁻) confirmed"));
        qualitativeReactions.Add(new QualitativeReaction("Ammonium Test", "NH₄⁺", "Sodium Hydroxide", "Ammonia gas evolved, pungent odor", "Ammonium ion (NH₄⁺) confirmed"));
    }
}
