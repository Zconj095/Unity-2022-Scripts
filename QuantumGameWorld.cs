using UnityEngine;
using System.Diagnostics;
using System.IO;

public class SystemsDesign10 : MonoBehaviour
{
    public GameObject targetGameObject;

    void Start()
    {
        string pythonInterpreter = "python";
        string pythonScriptPath = "Assets/Scripts/QuantumGameWorld.py";

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = pythonInterpreter;
        psi.Arguments = pythonScriptPath;
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        Process process = new Process();
        process.StartInfo = psi;
        process.Start();

        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();

        if (!string.IsNullOrEmpty(output))
        {
            UnityEngine.Debug.Log("Python script output: " + output);
        }
        if (!string.IsNullOrEmpty(error))
        {
            UnityEngine.Debug.LogError("Python script error: " + error);
        }

        process.WaitForExit();
        process.Close();

        // Read the output from the file
        string filePath = "Assets/Scripts/output.txt";
        if (File.Exists(filePath))
        {
            string fileContent = File.ReadAllText(filePath);
            UnityEngine.Debug.Log("File content: " + fileContent);

            // Example: Change the color of the target GameObject based on file content
            if (targetGameObject != null)
            {
                Renderer renderer = targetGameObject.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = Color.blue; // Change based on file content
                }
            }
        }
    }
}
