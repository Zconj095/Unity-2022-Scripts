using System.Collections.Generic;
using UnityEngine;

public class ElectrochemicalSeriesGenerator : MonoBehaviour
{
    [Header("Electrochemical Series")]
    public List<ElectrochemicalElement> electrochemicalSeries = new List<ElectrochemicalElement>();  // List to store the series

    void Start()
    {
        GenerateElectrochemicalSeries();
    }

    [ContextMenu("Generate Electrochemical Series")]
    public void GenerateElectrochemicalSeries()
    {
        electrochemicalSeries.Clear();  // Clear the list before generating

        // Add elements to the electrochemical series, from most reactive to least reactive (approximate reduction potentials)
        electrochemicalSeries.Add(new ElectrochemicalElement("Lithium (Li)", -3.04f));  // Most reactive (loses electrons easily)
        electrochemicalSeries.Add(new ElectrochemicalElement("Potassium (K)", -2.93f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Calcium (Ca)", -2.87f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Sodium (Na)", -2.71f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Magnesium (Mg)", -2.37f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Aluminum (Al)", -1.66f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Manganese (Mn)", -1.18f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Zinc (Zn)", -0.76f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Chromium (Cr)", -0.74f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Iron (Fe)", -0.44f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Cadmium (Cd)", -0.40f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Nickel (Ni)", -0.25f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Tin (Sn)", -0.14f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Lead (Pb)", -0.13f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Hydrogen (H)", 0.00f));  // Standard reference element
        electrochemicalSeries.Add(new ElectrochemicalElement("Antimony (Sb)", 0.15f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Bismuth (Bi)", 0.20f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Copper (Cu)", 0.34f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Iodine (I)", 0.54f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Silver (Ag)", 0.80f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Mercury (Hg)", 0.85f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Platinum (Pt)", 1.20f));  // Very unreactive (doesn't lose electrons easily)
        electrochemicalSeries.Add(new ElectrochemicalElement("Gold (Au)", 1.50f));  // Least reactive (very unreactive)

        // Non-metals and halogens:
        electrochemicalSeries.Add(new ElectrochemicalElement("Fluorine (F)", 2.87f));  // Highly reactive nonmetal (gains electrons easily)
        electrochemicalSeries.Add(new ElectrochemicalElement("Chlorine (Cl)", 1.36f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Bromine (Br)", 1.07f));
        electrochemicalSeries.Add(new ElectrochemicalElement("Iodine (I)", 0.54f));
    }
}
