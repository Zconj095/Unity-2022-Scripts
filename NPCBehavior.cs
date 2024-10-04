using UnityEngine;
using UnityEngine.AI;

public class NPCBehavior : MonoBehaviour
{
    public enum NPCState
    {
        Idle,
        Alert,
        Aggressive,
        Flee
    }

    public NPCState currentState;
    public float health = 100f;
    public float alertDistance = 10f;
    public float attackDistance = 3f;

    private NavMeshAgent agent;
    private Transform player;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        player = GameObject.FindWithTag("Player").transform;

        currentState = NPCState.Idle;
    }

    void Update()
    {
        switch (currentState)
        {
            case NPCState.Idle:
                IdleBehavior();
                break;
            case NPCState.Alert:
                AlertBehavior();
                break;
            case NPCState.Aggressive:
                AggressiveBehavior();
                break;
            case NPCState.Flee:
                FleeBehavior();
                break;
        }
    }

    void IdleBehavior()
    {
        if (Vector3.Distance(transform.position, player.position) < alertDistance)
        {
            currentState = NPCState.Alert;
        }
    }

    void AlertBehavior()
    {
        if (Vector3.Distance(transform.position, player.position) < attackDistance)
        {
            currentState = NPCState.Aggressive;
        }
        else if (Vector3.Distance(transform.position, player.position) > alertDistance)
        {
            currentState = NPCState.Idle;
        }
    }

    void AggressiveBehavior()
    {
        agent.SetDestination(player.position);

        if (Vector3.Distance(transform.position, player.position) <= attackDistance)
        {
            AttackPlayer();
        }

        if (health < 30f)
        {
            currentState = NPCState.Flee;
        }
    }

    void FleeBehavior()
    {
        Vector3 fleeDirection = transform.position - player.position;
        Vector3 newFleePosition = transform.position + fleeDirection;

        agent.SetDestination(newFleePosition);

        if (Vector3.Distance(transform.position, player.position) > alertDistance)
        {
            currentState = NPCState.Idle;
        }
    }

    void AttackPlayer()
    {
        // Implement attack logic here
        Debug.Log("Attacking the player!");
    }
}
