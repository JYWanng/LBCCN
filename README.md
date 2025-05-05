# A Lightweight and Real-Time Binaural Speech Enhancement Model with Spatial Cues Preservation

This is the official implementation of the LBCCN.

⚠️Note: The model is trained in a diffuse noise field with a fixed speaker direction (e.g., **55 degrees** to the right of the frontal direction). Each training instance of the model targets only one direction.  
The noise data used for training is from the **DEMAND** dataset.

---

## 🔧 Errata

We have identified and corrected the following inaccuracies in the original paper:

1. **Speaker Direction**:  
   The actual speaker direction used in the binaural speech enhancement experiments is **55 degrees**, not 45 degrees as previously mentioned.

2. **Noise Dataset**:  
   The noise data used for training is from the **DEMAND** dataset, **not NoiseX92** as may have been previously implied.

We apologize for any confusion this may have caused and appreciate your understanding.
