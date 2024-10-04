// C# script to execute a Python script in Unity

using UnityEngine;
using System.Diagnostics;

public class ASUNA_AI : MonoBehaviour
{
    void Start()
    {
        // Path to the Python interpreter and the Python script file
        string pythonInterpreter = "python";
        string pythonScriptPath = "Assets/Scripts/ARRAYS/LANGUAGE_TO_RUNIC/__init__.py";

        // Create process info
        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = pythonInterpreter;
        psi.Arguments = pythonScriptPath;
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        // Start the process
        Process process = new Process();
        process.StartInfo = psi;
        process.Start();

        // Read the output and error streams
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();

        // Log the output and error using UnityEngine.Debug
        if (!string.IsNullOrEmpty(output))
        {
            UnityEngine.Debug.Log("Python script output: " + output);
        }
        if (!string.IsNullOrEmpty(error))
        {
            UnityEngine.Debug.LogError("Python script error: " + error);
        }

        // Wait for the process to exit
        process.WaitForExit();
        process.Close();
    }
}
