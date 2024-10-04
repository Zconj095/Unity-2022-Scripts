using UnityEngine;
using UnityEngine.AI;

public class MythicalBeastBehavior : MonoBehaviour
{
    public NavMeshAgent agent;
    public Transform lair;
    public float roamRadius = 50f;
    public float detectionRange = 30f;
    public LayerMask targetLayer;

    private Vector3 roamPosition;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        Roam();
    }

    void Update()
    {
        DetectPlayer();

        if (!agent.pathPending && agent.remainingDistance < 0.5f)
        {
            Roam();
        }
    }

    void Roam()
    {
        roamPosition = lair.position + (Random.insideUnitSphere * roamRadius);
        roamPosition.y = lair.position.y;
        agent.SetDestination(roamPosition);
    }

    void DetectPlayer()
    {
        Collider[] targets = Physics.OverlapSphere(transform.position, detectionRange, targetLayer);
        if (targets.Length > 0)
        {
            agent.SetDestination(targets[0].transform.position);
            // Add more complex behavior here (e.g., attacking, fleeing)
        }
    }
}
