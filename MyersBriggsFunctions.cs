using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public enum CognitiveFunction
{
    ExtravertedSensing, // Se
    IntrovertedSensing, // Si
    ExtravertedIntuition, // Ne
    IntrovertedIntuition, // Ni
    ExtravertedThinking, // Te
    IntrovertedThinking, // Ti
    ExtravertedFeeling, // Fe
    IntrovertedFeeling // Fi
}

public enum PersonalityType
{
    ISTJ, ISFJ, INFJ, INTJ,
    ISTP, ISFP, INFP, INTP,
    ESTP, ESFP, ENFP, ENTP,
    ESTJ, ESFJ, ENFJ, ENTJ
}

public class MyersBriggsFunctions: MonoBehaviour
{
    private static readonly Dictionary<PersonalityType, CognitiveFunction[]> functionMap = new Dictionary<PersonalityType, CognitiveFunction[]>
    {
        { PersonalityType.ISTJ, new[] { CognitiveFunction.IntrovertedSensing, CognitiveFunction.ExtravertedThinking, CognitiveFunction.IntrovertedFeeling, CognitiveFunction.ExtravertedIntuition } },
        { PersonalityType.ISFJ, new[] { CognitiveFunction.IntrovertedSensing, CognitiveFunction.ExtravertedFeeling, CognitiveFunction.IntrovertedThinking, CognitiveFunction.ExtravertedIntuition } },
        { PersonalityType.INFJ, new[] { CognitiveFunction.IntrovertedIntuition, CognitiveFunction.ExtravertedFeeling, CognitiveFunction.IntrovertedThinking, CognitiveFunction.ExtravertedSensing } },
        { PersonalityType.INTJ, new[] { CognitiveFunction.IntrovertedIntuition, CognitiveFunction.ExtravertedThinking, CognitiveFunction.IntrovertedFeeling, CognitiveFunction.ExtravertedSensing } },

        { PersonalityType.ISTP, new[] { CognitiveFunction.IntrovertedThinking, CognitiveFunction.ExtravertedSensing, CognitiveFunction.IntrovertedIntuition, CognitiveFunction.ExtravertedFeeling } },
        { PersonalityType.ISFP, new[] { CognitiveFunction.IntrovertedFeeling, CognitiveFunction.ExtravertedSensing, CognitiveFunction.IntrovertedIntuition, CognitiveFunction.ExtravertedThinking } },
        { PersonalityType.INFP, new[] { CognitiveFunction.IntrovertedFeeling, CognitiveFunction.ExtravertedIntuition, CognitiveFunction.IntrovertedSensing, CognitiveFunction.ExtravertedThinking } },
        { PersonalityType.INTP, new[] { CognitiveFunction.IntrovertedThinking, CognitiveFunction.ExtravertedIntuition, CognitiveFunction.IntrovertedSensing, CognitiveFunction.ExtravertedFeeling } },

        { PersonalityType.ESTP, new[] { CognitiveFunction.ExtravertedSensing, CognitiveFunction.IntrovertedThinking, CognitiveFunction.ExtravertedFeeling, CognitiveFunction.IntrovertedIntuition } },
        { PersonalityType.ESFP, new[] { CognitiveFunction.ExtravertedSensing, CognitiveFunction.IntrovertedFeeling, CognitiveFunction.ExtravertedThinking, CognitiveFunction.IntrovertedIntuition } },
        { PersonalityType.ENFP, new[] { CognitiveFunction.ExtravertedIntuition, CognitiveFunction.IntrovertedFeeling, CognitiveFunction.ExtravertedThinking, CognitiveFunction.IntrovertedSensing } },
        { PersonalityType.ENTP, new[] { CognitiveFunction.ExtravertedIntuition, CognitiveFunction.IntrovertedThinking, CognitiveFunction.ExtravertedFeeling, CognitiveFunction.IntrovertedSensing } },

        { PersonalityType.ESTJ, new[] { CognitiveFunction.ExtravertedThinking, CognitiveFunction.IntrovertedSensing, CognitiveFunction.ExtravertedIntuition, CognitiveFunction.IntrovertedFeeling } },
        { PersonalityType.ESFJ, new[] { CognitiveFunction.ExtravertedFeeling, CognitiveFunction.IntrovertedSensing, CognitiveFunction.ExtravertedIntuition, CognitiveFunction.IntrovertedThinking } },
        { PersonalityType.ENFJ, new[] { CognitiveFunction.ExtravertedFeeling, CognitiveFunction.IntrovertedIntuition, CognitiveFunction.ExtravertedSensing, CognitiveFunction.IntrovertedThinking } },
        { PersonalityType.ENTJ, new[] { CognitiveFunction.ExtravertedThinking, CognitiveFunction.IntrovertedIntuition, CognitiveFunction.ExtravertedSensing, CognitiveFunction.IntrovertedFeeling } }
    };

    public static CognitiveFunction[] GetFunctionsForType(PersonalityType type)
    {
        return functionMap[type];
    }
}

public class CognitiveFunctionEditor : EditorWindow
{
    private PersonalityType selectedPersonality;
    private Vector2 scrollPos;

    [MenuItem("Window/Cognitive Function Viewer")]
    public static void ShowWindow()
    {
        GetWindow<CognitiveFunctionEditor>("Cognitive Function Viewer");
    }

    private void OnGUI()
    {
        GUILayout.Label("Select Personality Type", EditorStyles.boldLabel);

        // Dropdown to select personality type
        selectedPersonality = (PersonalityType)EditorGUILayout.EnumPopup("Personality Type", selectedPersonality);

        // Show the Cognitive Functions
        if (GUILayout.Button("Show Cognitive Functions"))
        {
            CognitiveFunction[] functions = MyersBriggsFunctions.GetFunctionsForType(selectedPersonality);

            GUILayout.Label("Cognitive Functions for " + selectedPersonality + ":");
            scrollPos = EditorGUILayout.BeginScrollView(scrollPos, GUILayout.Height(100));
            
            foreach (var function in functions)
            {
                GUILayout.Label(function.ToString());
            }

            EditorGUILayout.EndScrollView();
        }
    }
}
