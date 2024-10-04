using UnityEngine;

public class MineableObject : MonoBehaviour
{
    public int health = 10;
    public GameObject dropPrefab;

    public void Mine(int miningPower)
    {
        health -= miningPower;

        if (health <= 0)
        {
            DropLoot();
            Destroy(gameObject);
        }
    }

    void DropLoot()
    {
        Instantiate(dropPrefab, transform.position, Quaternion.identity);
    }
}
