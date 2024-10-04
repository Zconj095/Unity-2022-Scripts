using UnityEngine;

public class DifficultyScaler : MonoBehaviour
{
    public NPCBehavior npcBehavior;
    public int playerLevel = 1;

    void Start()
    {
        // Ensure NPCBehavior is assigned
        if (npcBehavior == null)
        {
            npcBehavior = GetComponent<NPCBehavior>();
            if (npcBehavior == null)
            {
                Debug.LogError("NPCBehavior component not found on the same GameObject!");
            }
        }
    }

    void Update()
    {
        ScaleDifficulty(playerLevel);
    }

    void ScaleDifficulty(int level)
    {
        if (npcBehavior != null)
        {
            npcBehavior.health = 100f + (level * 10f);
            npcBehavior.alertDistance = 10f + (level * 0.5f);
            npcBehavior.attackDistance = 3f + (level * 0.2f);
        }
    }
}
