using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
public class GlialCell: MonoBehaviour {
    public float Ca = 0.1f;

    public void Uptake(float glutamate) {
        Ca += 0.1f * glutamate;
    }

    public bool Gliotransmission() {
        if (Ca > 0.2f) {
            Ca -= 0.1f;
            return true;
        }
        return false;
    }
}