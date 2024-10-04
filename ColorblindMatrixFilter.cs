using UnityEngine;

public class ColorblindMatrixFilter : MonoBehaviour
{
    public Texture2D ApplyColorblindMatrixFilter(Texture2D original, string filterType)
    {
        // Define transformation matrices for different colorblindness types
        Matrix4x4 transformationMatrix = GetMatrix(filterType);
        
        Color32[] originalPixels = original.GetPixels32();
        Color32[] adjustedPixels = new Color32[originalPixels.Length];
        
        for (int i = 0; i < originalPixels.Length; i++)
        {
            Vector3 colorVec = new Vector3(originalPixels[i].r, originalPixels[i].g, originalPixels[i].b);
            colorVec = transformationMatrix.MultiplyPoint3x4(colorVec);
            
            // Ensure the RGB values are within the valid range
            colorVec.x = Mathf.Clamp(colorVec.x, 0, 255);
            colorVec.y = Mathf.Clamp(colorVec.y, 0, 255);
            colorVec.z = Mathf.Clamp(colorVec.z, 0, 255);

            adjustedPixels[i] = new Color32((byte)colorVec.x, (byte)colorVec.y, (byte)colorVec.z, originalPixels[i].a);
        }

        Texture2D adjustedImage = new Texture2D(original.width, original.height);
        adjustedImage.SetPixels32(adjustedPixels);
        adjustedImage.Apply();

        return adjustedImage;
    }

    private Matrix4x4 GetMatrix(string filterType)
    {
        switch (filterType)
        {
            case "protanopia":
                return Matrix4x4.identity; // Replace with actual matrix
            // add additional cases for other filter types
            default:
                return Matrix4x4.identity;
        }
    }
}