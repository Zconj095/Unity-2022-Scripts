using UnityEngine;

public class MeshHandler : MonoBehaviour
{
    public string meshID;
    private AnimationData currentAnimation;

    public MeshHandler(string id)
    {
        this.meshID = id;
    }

    public void ApplyAnimation(AnimationDatabase animationDatabase, string animationID)
    {
        currentAnimation = animationDatabase.RetrieveAnimation(animationID);
        if (currentAnimation != null)
        {
            Debug.Log($"Animation '{animationID}' applied to Mesh '{meshID}'.");
        }
    }

    public void PerformAnimation()
    {
        if (currentAnimation != null)
        {
            Debug.Log($"Mesh '{meshID}' is performing the animation with duration: {currentAnimation.duration} seconds.");
            // Placeholder for actual animation logic
        }
        else
        {
            Debug.LogWarning($"No animation is currently applied to Mesh '{meshID}'.");
        }
    }
}
