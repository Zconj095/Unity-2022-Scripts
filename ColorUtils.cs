using UnityEngine;

public class ColorUtils : MonoBehaviour
{
    public static Color HSVToRGB(float H, float S, float V)
    {
        if (S == 0f)
            return new Color(V, V, V);
        else if (V == 0f)
            return Color.black;
        else
        {
            Color col = Color.black;
            float Hval = H * 6f;
            int sel = Mathf.FloorToInt(Hval);
            float mod = Hval - sel;
            float v1 = V * (1f - S);
            float v2 = V * (1f - S * mod);
            float v3 = V * (1f - S * (1f - mod));
            switch (sel + 1)
            {
                case 0:
                    col.r = V; col.g = v1; col.b = v2; break;
                case 1:
                    col.r = V; col.g = v3; col.b = v1; break;
                case 2:
                    col.r = v2; col.g = V; col.b = v1; break;
                case 3:
                    col.r = v1; col.g = V; col.b = v3; break;
                case 4:
                    col.r = v1; col.g = v2; col.b = V; break;
                case 5:
                    col.r = v3; col.g = v1; col.b = V; break;
                case 6:
                    col.r = V; col.g = v1; col.b = v2; break;
                case 7:
                    col.r = V; col.g = v3; col.b = v1; break;
            }
            col.r = Mathf.Clamp01(col.r);
            col.g = Mathf.Clamp01(col.g);
            col.b = Mathf.Clamp01(col.b);
            return col;
        }
    }
}