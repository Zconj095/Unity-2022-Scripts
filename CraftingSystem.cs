using System.Collections.Generic;
using UnityEngine;

public class CraftingSystem : MonoBehaviour
{
    public List<CraftingRecipe> recipes;
    public Dictionary<string, int> playerResources = new Dictionary<string, int>();

    public void AddResource(string resource, int amount)
    {
        if (playerResources.ContainsKey(resource))
        {
            playerResources[resource] += amount;
        }
        else
        {
            playerResources.Add(resource, amount);
        }
    }

    public bool TryCraft(string itemName)
    {
        CraftingRecipe recipe = recipes.Find(r => r.itemName == itemName);

        if (recipe != null && recipe.CanCraft(playerResources))
        {
            recipe.Craft(playerResources);
            Instantiate(recipe.craftedItemPrefab, transform.position, Quaternion.identity);
            return true;
        }

        Debug.Log("Not enough resources to craft " + itemName);
        return false;
    }
}
