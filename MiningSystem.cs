using UnityEngine;

public class MiningSystem : MonoBehaviour
{
    public float miningRange = 2f;
    public LayerMask mineableLayer;
    public int baseMiningPower = 1;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Mine();
        }
    }

    void Mine()
    {
        RaycastHit hit;

        if (Physics.Raycast(Camera.main.ScreenPointToRay(Input.mousePosition), out hit, miningRange, mineableLayer))
        {
            MineableObject mineable = hit.collider.GetComponent<MineableObject>();

            if (mineable != null)
            {
                mineable.Mine(baseMiningPower);
            }
        }
    }
}
