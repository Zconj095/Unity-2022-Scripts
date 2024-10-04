using UnityEngine;
using System;

public class Synapse : MonoBehaviour
{
    public int numReceptors = 100;

    public void LTP()
    {
        numReceptors += 1;
    }

    public void LTD()
    {
        numReceptors -= 1;
    }
}