using UnityEngine;

public class Spirit : MonoBehaviour
{
    // Properties
    public bool IsIntangible = true;
    public bool IsInvisible = true;
    public float MovementSpeed = 2.0f;
    public bool IsHaunting = false;
    public string Intention = "Guide"; // Possible values: "Guide", "Revenge", "Enlightenment", etc.

    private Renderer spiritRenderer;

    void Start()
    {
        spiritRenderer = GetComponent<Renderer>();
        SetVisibility(IsInvisible);
    }

    void Update()
    {
        EtherealMovement();
        CheckTriggers();
    }

    // Method to handle the ethereal movement
    void EtherealMovement()
    {
        transform.Translate(Vector3.forward * MovementSpeed * Time.deltaTime);
    }

    // Method to handle visibility
    void SetVisibility(bool invisible)
    {
        if (invisible)
        {
            spiritRenderer.enabled = false;
        }
        else
        {
            spiritRenderer.enabled = true;
        }
    }

    // Method to check for triggers
    void CheckTriggers()
    {
        Collider[] hitColliders = Physics.OverlapSphere(transform.position, 5f);
        foreach (var hitCollider in hitColliders)
        {
            if (hitCollider.CompareTag("Player"))
            {
                InteractWithPlayer(hitCollider.gameObject);
            }
        }
    }

    // Method to handle haunting abilities
    public void Haunt()
    {
        if (IsHaunting)
        {
            // Example haunting effect: make nearby objects float
            Collider[] nearbyObjects = Physics.OverlapSphere(transform.position, 3f);
            foreach (var obj in nearbyObjects)
            {
                if (obj.GetComponent<Rigidbody>() != null)
                {
                    Rigidbody rb = obj.GetComponent<Rigidbody>();
                    rb.AddForce(Vector3.up * 5f, ForceMode.Impulse);
                }
            }
        }
    }

    // Interaction based on the spirit's intention or motivation
    public void InteractWithPlayer(GameObject player)
    {
        switch (Intention)
        {
            case "Guide":
                GuidePlayer(player);
                break;
            case "Revenge":
                BreakTheRules();
                break;
            case "Enlightenment":
                ShareWisdom();
                break;
                // Add other intentions as needed
        }
    }

    // Guide the player by providing clues or directions
    void GuidePlayer(GameObject player)
    {
        // Example: Highlight a path or object the player should follow
        Debug.Log("Guiding the player to their next objective.");
        // Implement more detailed guidance logic here
        // For instance, you can show a glowing path or an indicator towards the next objective
    }

    // Implement logic for revenge, like sabotaging player actions
    void BreakTheRules()
    {
        Debug.Log("The spirit is interfering with the player's actions!");
        // Example: Lock a door the player just opened or create an obstacle
        // Implement more complex sabotage logic here, such as altering the environment or disabling certain player abilities
    }

    // Share deep lore or knowledge with the player
    void ShareWisdom()
    {
        Debug.Log("The spirit shares ancient knowledge with the player.");
        // Example: Reveal hidden lore, provide critical hints for progression, or unlock a new ability
        // Implement more engaging methods of delivering wisdom, such as showing a cutscene, displaying text, or altering the player's perception temporarily
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            // Trigger the spirit to appear or perform an action
            SetVisibility(false);
            InteractWithPlayer(other.gameObject);
        }
    }

}
