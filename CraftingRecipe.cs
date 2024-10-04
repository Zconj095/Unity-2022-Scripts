using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class CraftingRecipe: MonoBehaviour
{
    public string itemName;
    public List<string> requiredGemstones;
    public List<int> requiredAmounts;
    public GameObject craftedItemPrefab;

    public bool CanCraft(Dictionary<string, int> playerResources)
    {
        for (int i = 0; i < requiredGemstones.Count; i++)
        {
            if (!playerResources.ContainsKey(requiredGemstones[i]) || playerResources[requiredGemstones[i]] < requiredAmounts[i])
            {
                return false;
            }
        }
        return true;
    }

    public void Craft(Dictionary<string, int> playerResources)
    {
        for (int i = 0; i < requiredGemstones.Count; i++)
        {
            playerResources[requiredGemstones[i]] -= requiredAmounts[i];
        }
    }
}
