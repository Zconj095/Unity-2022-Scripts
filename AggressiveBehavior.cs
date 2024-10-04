using UnityEngine;
using UnityEngine.AI;

public class AggressiveBehavior : MonoBehaviour
{
    // These fields will be visible in the Inspector
    [SerializeField] private NavMeshAgent agent;
    [SerializeField] private Transform player;
    [SerializeField] private float attackDistance;
    [SerializeField] private float health;
    [SerializeField] private NPCBehavior.NPCState currentState;

    // Initialize with necessary context (this can be called from another script)
    public void Initialize(NavMeshAgent navAgent, Transform playerTransform, float attackDist, float npcHealth, NPCBehavior.NPCState npcState)
    {
        agent = navAgent;
        player = playerTransform;
        attackDistance = attackDist;
        health = npcHealth;
        currentState = npcState;
    }

    public void ExecuteBehavior()
    {
        if (agent == null || player == null)
        {
            Debug.LogError("AggressiveBehavior not initialized properly.");
            return;
        }

        agent.SetDestination(player.position);

        if (Vector3.Distance(transform.position, player.position) <= attackDistance)
        {
            AttackPlayer();
        }

        if (health < 30f)
        {
            currentState = NPCBehavior.NPCState.Flee;
        }
        else if (Vector3.Distance(transform.position, player.position) > attackDistance)
        {
            currentState = NPCBehavior.NPCState.Alert;
        }
    }

    private void AttackPlayer()
    {
        if (IsRangedNPC())
        {
            Debug.Log("NPC uses ranged attack!");
        }
        else
        {
            Debug.Log("NPC uses melee attack!");
        }
    }

    private bool IsRangedNPC()
    {
        // Adjust this logic depending on your NPC type
        return false;
    }
}
