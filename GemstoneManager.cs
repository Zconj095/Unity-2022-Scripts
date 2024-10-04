using UnityEngine;
using System.Collections.Generic;

public class GemstoneManager : MonoBehaviour
{
    public static GemstoneManager Instance;

    [System.Serializable]
    public struct GemstoneData
    {
        public string gemstoneName;
        public bool isUnlocked;
    }

    public List<GemstoneData> gemstoneList = new List<GemstoneData>();

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public void EnableGemstone(string gemstoneName)
    {
        for (int i = 0; i < gemstoneList.Count; i++)
        {
            if (gemstoneList[i].gemstoneName == gemstoneName)
            {
                GemstoneData data = gemstoneList[i];
                data.isUnlocked = true;
                gemstoneList[i] = data;
                return;
            }
        }
    }

    public bool IsGemstoneUnlocked(string gemstoneName)
    {
        foreach (GemstoneData data in gemstoneList)
        {
            if (data.gemstoneName == gemstoneName)
            {
                return data.isUnlocked;
            }
        }
        return false;
    }
}
