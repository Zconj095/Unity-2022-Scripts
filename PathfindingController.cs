using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PathfindingController : MonoBehaviour
{
    [SerializeField]
    private float speed = 2.0f;

    protected float movementWaitTime = 0.5f;
    protected List<Vector3> path;
    private int pathIndex = 0;
    private Vector3 movementDestination;

    void Start()
    {
        path = GetPathFromPython();
        if (path != null && path.Count > 0)
        {
            movementDestination = path[pathIndex];
            StartCoroutine(MoveAlongPath());
        }
    }

    List<Vector3> GetPathFromPython()
    {
        List<Vector3> path = new List<Vector3>();
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = "pathfinding.py",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };

        using (Process process = Process.Start(start))
        {
            using (StreamReader reader = process.StandardOutput)
            {
                string result = reader.ReadToEnd();
                string[] points = result.Trim().Replace("Path:", "").Replace("(", "").Replace(")", "").Split(',');
                for (int i = 0; i < points.Length; i += 2)
                {
                    float x = float.Parse(points[i].Trim());
                    float y = float.Parse(points[i + 1].Trim());
                    path.Add(new Vector3(x, 0, y));
                }
            }
        }
        return path;
    }

    IEnumerator MoveAlongPath()
    {
        while (pathIndex < path.Count)
        {
            transform.position = Vector3.MoveTowards(transform.position, movementDestination, speed * Time.deltaTime);
            if (Vector3.Distance(transform.position, movementDestination) < 0.1f)
            {
                pathIndex++;
                if (pathIndex < path.Count)
                {
                    movementDestination = path[pathIndex];
                    yield return new WaitForSeconds(movementWaitTime);
                }
            }
            yield return null;
        }
    }
}
