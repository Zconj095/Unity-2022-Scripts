using System.Collections.Generic; // Add this line to include the List<T> class
using UnityEngine;

[CreateAssetMenu(fileName = "NewAnimationDatabase", menuName = "Animation/AnimationDatabase")]
public class AnimationDatabaseScriptable : ScriptableObject
{
    public List<AnimationData> animationList = new List<AnimationData>();
}
