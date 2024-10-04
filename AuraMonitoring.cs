using System;
using System.Collections.Generic;
using UnityEngine;
public class AuraMonitoring: MonoBehaviour{
    private Dictionary<string, List<float>> sensorData;  // Changed from generic object to specific data type

    public AuraMonitoring(Dictionary<string, List<float>> sensorData) {
        this.sensorData = sensorData;
    }

    public string AnalyzeAura() {
        // Logic for analyzing aura data could be added here.
        string auraAnalysis = "Aura analysis based on sensor inputs.";
        
        if (sensorData.ContainsKey("em_field_readings")) {
            auraAnalysis += $" Detected EM readings with {sensorData["em_field_readings"].Count} points."; 
            // Expand upon reading and processing actual values as needed.
        }
        
        return auraAnalysis;
    }
}

// Example usage:
public class ExampleUsage : MonoBehaviour {

    void Start() {
		Dictionary<string, List<float>> sensor_data = new Dictionary<string, List<float>>();
		sensor_data.Add("em_field_readings", new List<float>() {1.0f, 2.5f}); 
															 
		AuraMonitoring auraMonitor = new AuraMonitoring(sensor_data);
		
		Debug.Log(auraMonitor.AnalyzeAura());
	    }
}