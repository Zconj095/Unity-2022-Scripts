using UnityEngine;

namespace VisualsProcessing
{
    public class ColorblindFilter : MonoBehaviour
    {
        public Texture2D ApplyColorblindFilter(Texture2D original, string filterType)
        {
            Color32[] pixels = original.GetPixels32();
            float[,] matrix = GetMatrix(filterType);

            for (int i = 0; i < pixels.Length; i++)
            {
                Vector3 color = new Vector3(pixels[i].r, pixels[i].g, pixels[i].b);
                color = MultiplyMatrixVector(matrix, color);
                pixels[i] = new Color32((byte)Mathf.Clamp(color.x, 0, 255), (byte)Mathf.Clamp(color.y, 0, 255), (byte)Mathf.Clamp(color.z, 0, 255), pixels[i].a);
            }

            Texture2D adjustedImage = new Texture2D(original.width, original.height);
            adjustedImage.SetPixels32(pixels);
            adjustedImage.Apply();
            return adjustedImage;
        }

        private float[,] GetMatrix(string type)
        {
            // Define matrices here based on type
            // Sample implementation for protanopia (as a demonstration; replace with accurate data)
            switch (type)
            {
                case "protanopia":
                    return new float[,]
                    {
                        { 0.567f, 0.433f, 0f },
                        { 0.558f, 0.442f, 0f },
                        { 0f, 0.242f, 0.758f }
                    };
                default:
                    return new float[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };  // Identity matrix
            }
        }

        private Vector3 MultiplyMatrixVector(float[,] matrix, Vector3 vector)
        {
            Vector3 result = new Vector3(
                matrix[0, 0] * vector.x + matrix[0, 1] * vector.y + matrix[0, 2] * vector.z,
                matrix[1, 0] * vector.x + matrix[1, 1] * vector.y + matrix[1, 2] * vector.z,
                matrix[2, 0] * vector.x + matrix[2, 1] * vector.y + matrix[2, 2] * vector.z
            );
            return result;
        }
    }
}