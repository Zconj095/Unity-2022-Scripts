using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(ElectronConfigurationGenerator))]
public class ElectronConfigurationGeneratorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector(); // Draw the default inspector UI

        ElectronConfigurationGenerator generator = (ElectronConfigurationGenerator)target;

        if (GUILayout.Button("Generate Electron Configuration"))
        {
            generator.GenerateElectronConfiguration(generator.atomicNumber);
        }

        // Display the electron configuration in the Inspector
        if (generator.electronConfiguration.Count > 0)
        {
            GUILayout.Space(10);
            GUILayout.Label("Electron Configuration", EditorStyles.boldLabel);

            foreach (ElectronSubshell subshell in generator.electronConfiguration)
            {
                EditorGUILayout.LabelField($"{subshell.energyLevel}{subshell.orbitalType} : {subshell.maxElectrons} electrons");
            }
        }
    }
}
