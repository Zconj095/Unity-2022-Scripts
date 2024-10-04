using UnityEngine;

public class ChakraParticleBehavior : MonoBehaviour
{
    // Chakra layers
    public GameObject rootChakra;
    public GameObject sacralChakra;
    public GameObject solarPlexusChakra;
    public GameObject heartChakra;
    public GameObject throatChakra;
    public GameObject thirdEyeChakra;
    public GameObject crownChakra;

    // Moods, Emotions, Feelings as enums for easy assignment
    public enum Mood { CALM, CHEERFUL, CONTENT, DEFENSIVE, ENERGETIC, EXCITED, GRATEFUL, JOYFUL, LOUD, MELLOW, NARRATIVE, NEUTRAL, PEACEFUL, QUIET, RELAXED, TALKATIVE }
    public enum Emotion { COURAGE, FAITH, GRATITUDE, HAPPINESS, HOPE, JOY, LOVE, SERENE, SERENITY, SERIOUS, TEMPERANCE }
    public enum Feeling { ACCEPTANCE, ADMIRATION, AFFECTIONATE, ALTRUISTIC, AMUSED, BEAUTIFUL, BLESSED, BRAVE, CALM, CHEERFUL, COLLECTED, CONCERNED, CONFIDENT, CONTENT, COURAGEOUS, DEFENSIVE, DETERMINED, EAGER, EMOTIONAL, EMPATHIC, ENERGETIC, FAITH, FAITHFUL, FASCINATED, FOCUSED, FULL_OF_PURPOSE, GOOD, GRACEFUL, GRATEFUL, GREAT, HAPPY, HONOR, HONORABLE, HOPEFUL, IMPRESSED, INDEPENDENT, LIKED, LOVED, MOTIVATED, NEUTRAL, OFFENSIVE, OPTIMISTIC, POSITIVE, POWERFUL, PROUD, RELAXED, RELIEF, SELFLESS, SENSATIONAL, SENSITIVE, SOCIAL, SPECIAL, STRONG, SURPRISED, SYMPATHETIC, THANKFUL, THOUGHTFUL, THRILLED, UNIQUE }

    // Call this function to update particle behavior
    public void UpdateParticleBehavior(GameObject chakraLayer, Mood mood, Emotion emotion, Feeling feeling)
    {
        ParticleSystem particleSystem = chakraLayer.GetComponent<ParticleSystem>();
        if (particleSystem != null)
        {
            var main = particleSystem.main;

            // Dynamically adjust particle properties based on mood, emotion, and feeling
            AdjustByMood(particleSystem, mood);
            AdjustByEmotion(particleSystem, emotion);
            AdjustByFeeling(particleSystem, feeling);
        }
    }

    // Adjust particle behavior based on mood
    private void AdjustByMood(ParticleSystem particleSystem, Mood mood)
    {
        var main = particleSystem.main;
        switch (mood)
        {
            case Mood.CALM:
                main.startSize = 0.3f;
                main.simulationSpeed = 0.8f;
                main.startLifetime = 4.0f;
                break;

            case Mood.CHEERFUL:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                main.startLifetime = 5.0f;
                break;

            case Mood.CONTENT:
                main.startSize = 0.4f;
                main.simulationSpeed = 1.0f;
                main.startLifetime = 5.5f;
                break;

            case Mood.DEFENSIVE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.3f;
                main.startLifetime = 3.0f;
                break;

            case Mood.ENERGETIC:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.8f;
                main.startLifetime = 3.5f;
                break;

            case Mood.EXCITED:
                main.startSize = 1.0f;
                main.simulationSpeed = 2.0f;
                main.startLifetime = 2.0f;
                break;

            case Mood.GRATEFUL:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                main.startLifetime = 4.5f;
                break;

            case Mood.JOYFUL:
                main.startSize = 1.0f;
                main.simulationSpeed = 1.9f;
                main.startLifetime = 3.5f;
                break;

            case Mood.LOUD:
                main.startSize = 1.2f;
                main.simulationSpeed = 2.2f;
                main.startLifetime = 2.5f;
                break;

            case Mood.MELLOW:
                main.startSize = 0.5f;
                main.simulationSpeed = 0.9f;
                main.startLifetime = 5.5f;
                break;

            case Mood.NARRATIVE:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.3f;
                main.startLifetime = 4.0f;
                break;

            case Mood.NEUTRAL:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                main.startLifetime = 4.0f;
                break;

            case Mood.PEACEFUL:
                main.startSize = 0.4f;
                main.simulationSpeed = 0.7f;
                main.startLifetime = 6.0f;
                break;

            case Mood.QUIET:
                main.startSize = 0.3f;
                main.simulationSpeed = 0.6f;
                main.startLifetime = 5.0f;
                break;

            case Mood.RELAXED:
                main.startSize = 0.5f;
                main.simulationSpeed = 0.9f;
                main.startLifetime = 5.5f;
                break;

            case Mood.TALKATIVE:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.7f;
                main.startLifetime = 3.0f;
                break;

            default:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                main.startLifetime = 4.0f;
                break;
        }
    }


    // Adjust particle behavior based on emotion
    private void AdjustByEmotion(ParticleSystem particleSystem, Emotion emotion)
    {
        var main = particleSystem.main;
        var emission = particleSystem.emission;

        switch (emotion)
        {
            case Emotion.COURAGE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 40f;
                main.startLifetime = 4.0f;
                break;

            case Emotion.FAITH:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.5f;
                break;

            case Emotion.GRATITUDE:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;

            case Emotion.HAPPINESS:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.8f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.5f;
                break;

            case Emotion.HOPE:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.5f;
                break;

            case Emotion.JOY:
                main.startSize = 1.0f;
                main.simulationSpeed = 2.0f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.0f;
                break;

            case Emotion.LOVE:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.7f;
                emission.rateOverTime = 55f;
                main.startLifetime = 4.0f;
                break;

            case Emotion.SERENE:
                main.startSize = 0.4f;
                main.simulationSpeed = 0.8f;
                emission.rateOverTime = 25f;
                main.startLifetime = 6.0f;
                break;

            case Emotion.SERENITY:
                main.startSize = 0.3f;
                main.simulationSpeed = 0.7f;
                emission.rateOverTime = 20f;
                main.startLifetime = 6.5f;
                break;

            case Emotion.SERIOUS:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 30f;
                main.startLifetime = 5.0f;
                break;

            case Emotion.TEMPERANCE:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 28f;
                main.startLifetime = 5.5f;
                break;

            default:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;
        }
    }


    private void AdjustByFeeling(ParticleSystem particleSystem, Feeling feeling)
    {
        var main = particleSystem.main;
        var emission = particleSystem.emission;

        switch (feeling)
        {
            case Feeling.ACCEPTANCE:
                main.startSize = 0.5f;
                main.simulationSpeed = 0.9f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.ADMIRATION:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.1f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.AFFECTIONATE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.3f;
                emission.rateOverTime = 40f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.ALTRUISTIC:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 38f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.AMUSED:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.BEAUTIFUL:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.BLESSED:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 5.0f;
                break;

            case Feeling.BRAVE:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.CALM:
                main.startSize = 0.4f;
                main.simulationSpeed = 0.8f;
                emission.rateOverTime = 25f;
                main.startLifetime = 6.0f;
                break;

            case Feeling.CHEERFUL:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.COLLECTED:
                main.startSize = 0.5f;
                main.simulationSpeed = 0.9f;
                emission.rateOverTime = 28f;
                main.startLifetime = 5.5f;
                break;

            case Feeling.CONCERNED:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.CONFIDENT:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.CONTENT:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 5.0f;
                break;

            case Feeling.COURAGEOUS:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 48f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.DEFENSIVE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 40f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.DETERMINED:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.7f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.EAGER:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.3f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.EMOTIONAL:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 50f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.EMPATHIC:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 35f;
                main.startLifetime = 5.0f;
                break;

            case Feeling.ENERGETIC:
                main.startSize = 1.0f;
                main.simulationSpeed = 2.0f;
                emission.rateOverTime = 55f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.FAITH:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.3f;
                emission.rateOverTime = 40f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.FAITHFUL:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.3f;
                emission.rateOverTime = 40f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.FASCINATED:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.FOCUSED:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.FULL_OF_PURPOSE:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.GOOD:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.GRACEFUL:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.1f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.GRATEFUL:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.GREAT:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.HAPPY:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.HONOR:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.HONORABLE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 40f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.HOPEFUL:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.3f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.IMPRESSED:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.INDEPENDENT:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.LIKED:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.1f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.LOVED:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.7f;
                emission.rateOverTime = 50f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.MOTIVATED:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.8f;
                emission.rateOverTime = 55f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.NEUTRAL:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.OFFENSIVE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.OPTIMISTIC:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 40f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.POSITIVE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.3f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.POWERFUL:
                main.startSize = 1.0f;
                main.simulationSpeed = 2.0f;
                emission.rateOverTime = 55f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.PROUD:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.6f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.RELAXED:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 5.0f;
                break;

            case Feeling.RELIEF:
                main.startSize = 0.4f;
                main.simulationSpeed = 0.9f;
                emission.rateOverTime = 28f;
                main.startLifetime = 5.5f;
                break;

            case Feeling.SELFLESS:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.SENSATIONAL:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.8f;
                emission.rateOverTime = 55f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.SENSITIVE:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.1f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.SOCIAL:
                main.startSize = 0.8f;
                main.simulationSpeed = 1.5f;
                emission.rateOverTime = 45f;
                main.startLifetime = 3.5f;
                break;

            case Feeling.SPECIAL:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 40f;
                main.startLifetime = 4.0f;
                break;

            case Feeling.STRONG:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.8f;
                emission.rateOverTime = 50f;
                main.startLifetime = 3.0f;
                break;

            case Feeling.SURPRISED:
                main.startSize = 1.0f;
                main.simulationSpeed = 2.0f;
                emission.rateOverTime = 55f;
                main.startLifetime = 2.5f;
                break;

            case Feeling.SYMPATHETIC:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.1f;
                emission.rateOverTime = 30f;
                main.startLifetime = 5.0f;
                break;

            case Feeling.THANKFUL:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 35f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.THOUGHTFUL:
                main.startSize = 0.6f;
                main.simulationSpeed = 1.2f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.5f;
                break;

            case Feeling.THRILLED:
                main.startSize = 0.9f;
                main.simulationSpeed = 1.9f;
                emission.rateOverTime = 55f;
                main.startLifetime = 2.5f;
                break;

            case Feeling.UNIQUE:
                main.startSize = 0.7f;
                main.simulationSpeed = 1.4f;
                emission.rateOverTime = 40f;
                main.startLifetime = 3.5f;
                break;

            default:
                main.startSize = 0.5f;
                main.simulationSpeed = 1.0f;
                emission.rateOverTime = 30f;
                main.startLifetime = 4.0f;
                break;
        }
    }


    // Example to apply changes based on Chakra, Mood, Emotion, and Feeling
    public void ApplyChakraParticleBehavior(Mood mood, Emotion emotion, Feeling feeling)
    {
        // Apply for all chakra layers as example, you can modify for specific chakras
        UpdateParticleBehavior(rootChakra, mood, emotion, feeling);
        UpdateParticleBehavior(sacralChakra, mood, emotion, feeling);
        UpdateParticleBehavior(solarPlexusChakra, mood, emotion, feeling);
        UpdateParticleBehavior(heartChakra, mood, emotion, feeling);
        UpdateParticleBehavior(throatChakra, mood, emotion, feeling);
        UpdateParticleBehavior(thirdEyeChakra, mood, emotion, feeling);
        UpdateParticleBehavior(crownChakra, mood, emotion, feeling);
    }
}
