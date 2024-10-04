using UnityEngine;

public class ChakraSystem : MonoBehaviour
{
    public GameObject[] chakras; // Assign in Inspector: 0 - Root, 1 - Sacral, ..., 6 - Crown
    public float[] chakraEnergies; // Chakra energy levels for each chakra
    public float maxEnergy = 100f; // Max energy each chakra can have
    public float energyFlowSpeed = 5f; // Speed at which energy flows between chakras
    public float healingRate = 5f; // Health per second when Heart Chakra is active

    private bool heartChakraActive = false;
    private PlayerHealth playerHealth; // Assume you have a PlayerHealth script

    public AudioClip chakraActivationSound; // Optional: sound for chakra activation
    public AudioClip chakraDeactivationSound; // Optional: sound for chakra deactivation
    private AudioSource audioSource;

    void Start()
    {
        // Initialize chakras and player health
        foreach (GameObject chakra in chakras)
        {
            chakra.SetActive(true); // Activate all chakras initially
        }

        // Initialize chakra energies
        chakraEnergies = new float[chakras.Length];
        for (int i = 0; i < chakraEnergies.Length; i++)
        {
            chakraEnergies[i] = maxEnergy / 2; // Set each chakra to half energy as a starting point
        }

        playerHealth = GetComponent<PlayerHealth>(); // Assuming the PlayerHealth script is attached to the same GameObject
        audioSource = GetComponent<AudioSource>();
    }

    void Update()
    {
        // Healing logic: Heal the player if Heart Chakra is active
        if (heartChakraActive && playerHealth != null)
        {
            playerHealth.Heal(healingRate * Time.deltaTime); // Heal the player at the defined rate per second
        }

        // Example: Activate or deactivate chakras based on player input or conditions
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            ToggleChakra(0); // Toggle Root Chakra
        }
        else if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            ActivateHeartChakra(!heartChakraActive); // Toggle Heart Chakra
        }

        // Energy flow simulation
        BalanceChakras();
    }

    public void ToggleChakra(int index)
    {
        if (index >= 0 && index < chakras.Length)
        {
            bool isActive = chakras[index].activeSelf;
            chakras[index].SetActive(!isActive);

            // Play corresponding sound effect
            if (audioSource != null)
            {
                if (chakras[index].activeSelf)
                {
                    audioSource.PlayOneShot(chakraActivationSound);
                }
                else
                {
                    audioSource.PlayOneShot(chakraDeactivationSound);
                }
            }
        }
    }

    // Method to activate or deactivate the Heart Chakra and control healing
    public void ActivateHeartChakra(bool activate)
    {
        heartChakraActive = activate;
        chakras[3].SetActive(activate); // Heart Chakra index is 3 (assuming the chakras are ordered correctly in the array)

        // Play corresponding sound effect if needed
        if (audioSource != null)
        {
            if (activate)
            {
                audioSource.PlayOneShot(chakraActivationSound);
            }
            else
            {
                audioSource.PlayOneShot(chakraDeactivationSound);
            }
        }
    }

    void BalanceChakras()
    {
        // Example of energy flowing between chakras dynamically
        for (int i = 0; i < chakraEnergies.Length - 1; i++)
        {
            // Energy flow from lower to higher chakras (upward energy flow)
            float energyDifference = chakraEnergies[i] - chakraEnergies[i + 1];
            float energyFlow = energyDifference * Time.deltaTime * energyFlowSpeed;

            chakraEnergies[i] -= energyFlow;
            chakraEnergies[i + 1] += energyFlow;
        }

        // Ensure chakras don't exceed max or fall below zero energy
        for (int i = 0; i < chakraEnergies.Length; i++)
        {
            chakraEnergies[i] = Mathf.Clamp(chakraEnergies[i], 0, maxEnergy);
        }
    }

    public void AddEnergyToChakra(int index, float amount)
    {
        if (index >= 0 && index < chakras.Length)
        {
            chakraEnergies[index] += amount;
            chakraEnergies[index] = Mathf.Clamp(chakraEnergies[index], 0, maxEnergy);
        }
    }

    public float GetChakraEnergy(int index)
    {
        if (index >= 0 && index < chakras.Length)
        {
            return chakraEnergies[index];
        }
        return 0;
    }

    public void AlignChakras()
    {
        // Align all chakras by averaging their energies
        float averageEnergy = 0f;
        foreach (float energy in chakraEnergies)
        {
            averageEnergy += energy;
        }
        averageEnergy /= chakraEnergies.Length;

        for (int i = 0; i < chakraEnergies.Length; i++)
        {
            chakraEnergies[i] = averageEnergy;
        }
    }
}
