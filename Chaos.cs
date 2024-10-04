using UnityEngine;

[System.Serializable]
public class Chaos
{
    [SerializeField] private bool isWhiteChaos;

    public bool IsWhiteChaos => isWhiteChaos;

    public Chaos(bool isWhiteChaos)
    {
        this.isWhiteChaos = isWhiteChaos;
    }

    public void EnterChaosState()
    {
        if (isWhiteChaos)
        {
            Debug.Log("Entered White Chaos: Open to change and exploration.");
        }
        else
        {
            Debug.Log("Entered Black Chaos: Closed to change, filled with confusion.");
        }
    }
}
