using UnityEngine;

public class MineralGathering : MonoBehaviour
{
    public int resourceAmount = 100;

    public void GatherResource(int amount)
    {
        resourceAmount -= amount;
        if (resourceAmount <= 0)
        {
            Destroy(gameObject);
        }
    }
}
