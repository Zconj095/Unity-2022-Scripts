using UnityEngine;
using UnityEngine.UI;

public class UIAnimationController : MonoBehaviour
{
    public AnimationDatabase animationDatabase;
    public MeshHandler characterMesh;
    public Dropdown animationDropdown;
    
    void Start()
    {
        // Populate the dropdown with animation names
        PopulateDropdown();
        
        // Add a listener for when the dropdown value changes
        animationDropdown.onValueChanged.AddListener(delegate { DropdownValueChanged(animationDropdown); });
    }
    
    void PopulateDropdown()
    {
        foreach (AnimationData anim in animationDatabase.animationList)
        {
            animationDropdown.options.Add(new Dropdown.OptionData(anim.animationID));
        }
    }
    
    void DropdownValueChanged(Dropdown change)
    {
        string selectedAnimation = change.options[change.value].text;
        characterMesh.ApplyAnimation(animationDatabase, selectedAnimation);
        characterMesh.PerformAnimation();
    }
}
