using UnityEngine;
using System.Collections;
public class PlantGrowth : MonoBehaviour
{
    public float growthDuration = 60f; // Time in seconds for full growth
    private Vector3 initialScale;
    private Vector3 targetScale;

    void Start()
    {
        initialScale = transform.localScale;
        targetScale = initialScale * 2f; // Example: plant doubles in size
        transform.localScale = Vector3.zero;
        StartCoroutine(GrowPlant());
    }

    IEnumerator GrowPlant()
    {
        float elapsedTime = 0f;
        while (elapsedTime < growthDuration)
        {
            transform.localScale = Vector3.Lerp(Vector3.zero, targetScale, elapsedTime / growthDuration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        transform.localScale = targetScale;
    }
}
