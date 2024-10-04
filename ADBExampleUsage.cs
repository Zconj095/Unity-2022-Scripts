using System.Collections.Generic;
using UnityEngine;
public class ADBExampleUsage : MonoBehaviour
{
    public AnimationDatabase animationDatabase;
    private MeshHandler characterMesh;

    void Start()
    {
        // Store some animations in the database
        animationDatabase.StoreAnimation("walk_cycle", new List<int> { 0, 10, 20 }, 2.0f);
        animationDatabase.StoreAnimation("jump", new List<int> { 0, 15, 30 }, 1.0f);

        // Create a mesh and apply animations
        characterMesh = new MeshHandler("character_001");
        characterMesh.ApplyAnimation(animationDatabase, "walk_cycle");
        characterMesh.PerformAnimation();

        // Apply a different animation
        characterMesh.ApplyAnimation(animationDatabase, "jump");
        characterMesh.PerformAnimation();
    }
}
