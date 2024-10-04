using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerHealth : MonoBehaviour
{
    public float maxHealth = 100f; // Maximum health
    private float currentHealth; // Current health

    void Start()
    {
        currentHealth = maxHealth; // Initialize player with full health
    }

    public void Heal(float amount)
    {
        currentHealth += amount;
        currentHealth = Mathf.Clamp(currentHealth, 0f, maxHealth); // Ensure health doesn't exceed maxHealth
        Debug.Log("Player healed. Current health: " + currentHealth);
    }

    public void TakeDamage(float amount)
    {
        currentHealth -= amount;
        currentHealth = Mathf.Clamp(currentHealth, 0f, maxHealth); // Ensure health doesn't drop below zero
        Debug.Log("Player took damage. Current health: " + currentHealth);

        if (currentHealth <= 0f)
        {
            Die();
        }
    }

    void Die()
    {
        Debug.Log("Player died!");
        // Add logic for player death (e.g., respawn or game over)
    }
}
