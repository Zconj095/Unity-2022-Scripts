using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class AnimationData
{
    public string animationID;
    public List<int> keyframeData;
    public float duration;

    public AnimationData(string id, List<int> keyframes, float duration)
    {
        this.animationID = id;
        this.keyframeData = keyframes;
        this.duration = duration;
    }
}

public class AnimationDatabase : MonoBehaviour
{
    public List<AnimationData> animationList = new List<AnimationData>();

    void Start()
    {
        // Initialize with some default animations or ensure the list is ready to be used.
        if (animationList == null)
        {
            animationList = new List<AnimationData>();
        }
    }

    public void StoreAnimation(string animationID, List<int> keyframeData, float duration)
    {
        if (animationList == null)
        {
            animationList = new List<AnimationData>();
        }

        AnimationData animationData = new AnimationData(animationID, keyframeData, duration);
        animationList.Add(animationData);
        Debug.Log($"Animation '{animationID}' stored in the database.");
    }

    public AnimationData RetrieveAnimation(string animationID)
    {
        if (animationList != null)
        {
            foreach (AnimationData anim in animationList)
            {
                if (anim.animationID == animationID)
                {
                    Debug.Log($"Animation '{animationID}' retrieved.");
                    return anim;
                }
            }
        }

        Debug.LogWarning($"Animation '{animationID}' not found in the database.");
        return null;
    }
}
