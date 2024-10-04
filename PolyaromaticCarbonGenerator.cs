using System.Collections.Generic;
using UnityEngine;

public class PolyaromaticCarbonGenerator : MonoBehaviour
{
    [Header("Properties of Polyaromatic Hydrocarbons (PAHs)")]
    public List<PolyaromaticCarbon> polyaromaticCarbons = new List<PolyaromaticCarbon>();

    void Start()
    {
        GeneratePolyaromaticCarbons();
    }

    [ContextMenu("Generate Polyaromatic Carbons")]
    public void GeneratePolyaromaticCarbons()
    {
        polyaromaticCarbons.Clear();  // Clear the list before generating

        // Add common PAHs and their properties (name, molecular formula, molecular weight, number of rings, melting point, boiling point)
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Benzene", "C6H6", 78.11f, 1, 5.5f, 80.1f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Naphthalene", "C10H8", 128.17f, 2, 80.3f, 218.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Anthracene", "C14H10", 178.23f, 3, 216.0f, 340.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Phenanthrene", "C14H10", 178.23f, 3, 101.0f, 340.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Pyrene", "C16H10", 202.26f, 4, 150.0f, 404.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Chrysene", "C18H12", 228.29f, 4, 255.0f, 448.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Benzo[a]pyrene", "C20H12", 252.31f, 5, 179.0f, 495.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Coronene", "C24H12", 300.36f, 7, 438.0f, 525.0f));

        // Adding more known polyaromatic hydrocarbons
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Fluoranthene", "C16H10", 202.26f, 4, 110.0f, 375.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Benzo[e]pyrene", "C20H12", 252.31f, 5, 178.0f, 480.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Benzo[ghi]perylene", "C22H12", 276.33f, 6, 278.0f, 500.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Dibenz[a,h]anthracene", "C22H14", 278.35f, 5, 266.0f, 524.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Indeno[1,2,3-cd]pyrene", "C22H12", 276.33f, 6, 163.0f, 536.0f));
        polyaromaticCarbons.Add(new PolyaromaticCarbon("Perylene", "C20H12", 252.31f, 5, 278.0f, 500.0f));

    }
}
