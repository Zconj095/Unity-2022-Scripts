using UnityEngine;

public class HauntingEffect : MonoBehaviour
{
    public ParticleSystem hauntingParticles;
    public AudioClip hauntingSound;
    private AudioSource audioSource;

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
    }

    public void PlayHauntingEffect()
    {
        hauntingParticles.Play();
        audioSource.PlayOneShot(hauntingSound);
        // Add more haunting behavior like moving objects
    }
}

