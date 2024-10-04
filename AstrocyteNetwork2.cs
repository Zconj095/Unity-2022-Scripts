using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
public class AstrocyteNetwork2 : MonoBehaviour {
    public List<GlialCell> astrocytes;

    void Awake() {
        astrocytes = new List<GlialCell>();
        for (int i = 0; i < 20; i++) {
            astrocytes.Add(new GlialCell());
        }
    }

    public List<GlialCell> Simulate(float[] glutamate, int timesteps) {
        List<GlialCell> transmissions = new List<GlialCell>();

        for (int t = 0; t < timesteps; t++) {
            float glut = glutamate[Random.Range(0, glutamate.Length)];
            GlialCell astro = astrocytes[Random.Range(0, astrocytes.Count)];
            astro.Uptake(glut);

            if (astro.Gliotransmission()) {
                transmissions.Add(astro);
            }
        }

        return transmissions.Count > 5 ? transmissions.GetRange(transmissions.Count - 5, 5) : new List<GlialCell>();
    }
}