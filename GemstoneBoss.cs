using UnityEngine;

public class GemstoneBoss : MonoBehaviour
{
    public string gemstoneType;
    public int health = 100;
    public GameObject gemstonePrefab;
    public Transform spawnLocation;

    public void TakeDamage(int damage)
    {
        health -= damage;

        if (health <= 0)
        {
            OnBossDefeated();
        }
    }

    void OnBossDefeated()
    {
        // Spawn gemstone at specific location
        Instantiate(gemstonePrefab, spawnLocation.position, Quaternion.identity);

        // Additional logic for enabling gemstone in the mine
        GemstoneManager.Instance.EnableGemstone(gemstoneType);

        Destroy(gameObject);
    }
}
