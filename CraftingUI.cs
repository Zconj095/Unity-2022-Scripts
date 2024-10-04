using UnityEngine;
using UnityEngine.UI;

public class CraftingUI : MonoBehaviour
{
    public CraftingSystem craftingSystem;
    public GameObject craftingButtonPrefab;
    public Transform craftingMenu;

    void Start()
    {
        foreach (CraftingRecipe recipe in craftingSystem.recipes)
        {
            GameObject button = Instantiate(craftingButtonPrefab, craftingMenu);
            button.GetComponentInChildren<Text>().text = recipe.itemName;
            button.GetComponent<Button>().onClick.AddListener(() => CraftItem(recipe.itemName));
        }
    }

    void CraftItem(string itemName)
    {
        if (craftingSystem.TryCraft(itemName))
        {
            Debug.Log("Crafted " + itemName);
        }
    }
}
