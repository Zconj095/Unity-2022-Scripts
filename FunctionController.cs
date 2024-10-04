using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;

public class FunctionController : MonoBehaviour
{
    [SerializeField]
    private float speed = 2.0f;
    [SerializeField]
    private float rotationSpeed = 5.0f;
    [SerializeField]
    private float stoppingDistance = 0.5f;

    protected float movementWaitTime = 0.5f;

    void Start()
    {
        // Example usage of the functions
        UnityEngine.Debug.Log(SERIALIZEFIELD("Example Field"));
        UnityEngine.Debug.Log(PATHFINDINGMOVEMENT(new Vector2(0, 0), new Vector2(6, 6), GetGrid()));
        UnityEngine.Debug.Log(PROTECTEDFLOAT(0.5f));
        MOVEMENTWAITTIME(movementWaitTime);
        UnityEngine.Debug.Log(MOVEMENTDESTINATION(new List<Vector3> { new Vector3(1, 0, 1) }, 0));
        UnityEngine.Debug.Log(MOVEMENTCENTER(new List<Vector3> { new Vector3(1, 0, 1), new Vector3(2, 0, 2) }));
        UnityEngine.Debug.Log(DELTAROTATION(0f, 90f));
        UnityEngine.Debug.Log(INPUTVECTOR(1f, 1f));
        UnityEngine.Debug.Log(ISATDESTINATION(new Vector3(1, 0, 1), new Vector3(1, 0, 1)));
        UnityEngine.Debug.Log(MINFLOAT());
        UnityEngine.Debug.Log(MAXFLOAT());
        UnityEngine.Debug.Log(HASARRIVED(new Vector3(1, 0, 1), new Vector3(1, 0, 1), 0.1f));
        UnityEngine.Debug.Log(AGENTMOVEMENT(new Vector3(1, 0, 1), new Vector3(1, 0, 1), 1f));
        UnityEngine.Debug.Log(MOVEMENTSTOPPINGDISTANCE(10f, 2f));
        UnityEngine.Debug.Log(CHARACTERABILITIES("Dash"));
        UnityEngine.Debug.Log(MOVEMENTRANDOMWEIGHT(0.5f, 1.5f));
    }

    string SERIALIZEFIELD(string value)
    {
        return ExecutePythonFunction("SERIALIZEFIELD", value);
    }

    string PATHFINDINGMOVEMENT(Vector2 start, Vector2 end, int[,] grid)
    {
        string gridStr = GridToString(grid);
        return ExecutePythonFunction("PATHFINDINGMOVEMENT", $"{start.x} {start.y} {end.x} {end.y} {gridStr}");
    }

    string PROTECTEDFLOAT(float value)
    {
        return ExecutePythonFunction("PROTECTEDFLOAT", value.ToString());
    }

    void MOVEMENTWAITTIME(float waitTime)
    {
        ExecutePythonFunction("MOVEMENTWAITTIME", waitTime.ToString());
    }

    string MOVEMENTDESTINATION(List<Vector3> path, int index)
    {
        string pathStr = PathToString(path);
        return ExecutePythonFunction("MOVEMENTDESTINATION", $"{pathStr} {index}");
    }

    string MOVEMENTCENTER(List<Vector3> path)
    {
        string pathStr = PathToString(path);
        return ExecutePythonFunction("MOVEMENTCENTER", pathStr);
    }

    string DELTAROTATION(float currentRotation, float targetRotation)
    {
        return ExecutePythonFunction("DELTAROTATION", $"{currentRotation} {targetRotation}");
    }

    string INPUTVECTOR(float inputX, float inputY)
    {
        return ExecutePythonFunction("INPUTVECTOR", $"{inputX} {inputY}");
    }

    string ISATDESTINATION(Vector3 currentPosition, Vector3 destination)
    {
        return ExecutePythonFunction("ISATDESTINATION", $"{currentPosition.x} {currentPosition.y} {currentPosition.z} {destination.x} {destination.y} {destination.z}");
    }

    string MINFLOAT()
    {
        return ExecutePythonFunction("MINFLOAT", "");
    }

    string MAXFLOAT()
    {
        return ExecutePythonFunction("MAXFLOAT", "");
    }

    string HASARRIVED(Vector3 currentPosition, Vector3 destination, float stoppingDistance)
    {
        return ExecutePythonFunction("HASARRIVED", $"{currentPosition.x} {currentPosition.y} {currentPosition.z} {destination.x} {destination.y} {destination.z} {stoppingDistance}");
    }

    string AGENTMOVEMENT(Vector3 position, Vector3 velocity, float deltaTime)
    {
        return ExecutePythonFunction("AGENTMOVEMENT", $"{position.x} {position.y} {position.z} {velocity.x} {velocity.y} {velocity.z} {deltaTime}");
    }

    string MOVEMENTSTOPPINGDISTANCE(float velocity, float deceleration)
    {
        return ExecutePythonFunction("MOVEMENTSTOPPINGDISTANCE", $"{velocity} {deceleration}");
    }

    string CHARACTERABILITIES(string abilityName)
    {
        return ExecutePythonFunction("CHARACTERABILITIES", abilityName);
    }

    string MOVEMENTRANDOMWEIGHT(float minWeight, float maxWeight)
    {
        return ExecutePythonFunction("MOVEMENTRANDOMWEIGHT", $"{minWeight} {maxWeight}");
    }

    string ExecutePythonFunction(string functionName, string arguments)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = $"function_library.py {functionName} {arguments}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };

        using (Process process = Process.Start(startInfo))
        {
            using (StreamReader reader = process.StandardOutput)
            {
                return reader.ReadToEnd().Trim();
            }
        }
    }

    string GridToString(int[,] grid)
    {
        List<string> gridList = new List<string>();
        for (int i = 0; i < grid.GetLength(0); i++)
        {
            for (int j = 0; j < grid.GetLength(1); j++)
            {
                gridList.Add(grid[i, j].ToString());
            }
        }
        return string.Join(" ", gridList);
    }

    string PathToString(List<Vector3> path)
    {
        List<string> pathList = new List<string>();
        foreach (Vector3 point in path)
        {
            pathList.Add($"{point.x},{point.z}");
        }
        return string.Join(" ", pathList);
    }

    int[,] GetGrid()
    {
        return new int[,]
        {
            {0, 1, 0, 0, 0, 0, 0},
            {0, 1, 0, 1, 1, 1, 0},
            {0, 0, 0, 1, 0, 0, 0},
            {0, 1, 0, 1, 0, 1, 0},
            {0, 1, 0, 0, 0, 1, 0},
            {0, 1, 1, 1, 1, 1, 0},
            {0, 0, 0, 0, 0, 0, 0},
        };
    }
}
