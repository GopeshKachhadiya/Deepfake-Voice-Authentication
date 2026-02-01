# Deepfake vs. Human Classification Parameters

This document outlines the specific parameters and features used by the **Sentinel Neural Orchestrator** to classify audio as either Human or Deepfake.

## 1. Multi-Dimensional Feature Extraction

The system utilizes two distinct "expert" branches to analyze voice signals, focusing on different acoustic properties.

### A. Spectral Analysis (The 'Acoustic' Brain)
*   **Parameters**: Deep learning embeddings extracted from a pre-trained **Wav2Vec2** transformer model.
*   **Extraction Method**: The audio signal is processed through the Wav2Vec2 model, and the mean of the `last_hidden_state` (temporal average) is used as the feature vector.
*   **Purpose**: This captures high-level acoustic signatures, vocoder artifacts, and subtle spectral inconsistencies that are characteristic of AI-generated speech but imperceptible to the human ear.
*   **Classifier**: Random Forest.

### B. Prosody Analysis (The 'Rhythmic' Brain)
*   **Parameters**: 13 **MFCCs** (Mel-frequency cepstral coefficients).
*   **Extraction Method**: A sequence of 13 MFCCs is extracted across the time domain using `librosa`.
*   **Purpose**: Focuses on the "prosody" of speech—rhythm, timing, stress, and intonation. Deepfakes often struggle with consistent biometric rhythms or natural vocal tract dynamics, which this LSTM-based expert identifies.
*   **Classifier**: Bidirectional LSTM (Long Short-Term Memory) Neural Network.

---

## 2. Decision Logic & Arbitration

The **Sentinel Orchestrator** does not rely on a single score but rather "arbitrates" between the experts using the following logic:

### Weighted Fusion
The final risk score is generally calculated as a weighted average:
*   **Spectral Weight**: 65% (Primary influence due to high accuracy in detecting synthetic textures).
*   **Prosody Weight**: 35% (Secondary influence used to verify natural cadence).

### Critical Override
*   **Threshold**: 0.85 (Spectral)
*   **Logic**: If the Spectral Expert detects a risk higher than 85%, it triggers a "Critical Vocoder Artifact" alert and overrides the combined calculation. This ensures that even if a deepfake has "perfect rhythm," a high detection of synthetic artifacts will trigger an alarm.

### Temporal Memory Consolidation
*   **Window**: Last 3-5 audio chunks.
*   **Mechanism**: The system maintain a "Temporal Memory" (moving average) of recent risk scores.
*   **Benefit**: This prevents "flickering" results and reduces false positives caused by brief environmental noise or audio glitches, requiring a sustained "deepfake signature" to change the status to 🚨.

---

## 3. Data Augmentation for Robustness
During training, the system uses "Robustness Augmentation" to ensure these parameters work under real-world conditions:
*   **Telephony Filter**: Simulates 300Hz - 3400Hz bandpass (standard phone call quality).
*   **Gaussian Noise**: Adds white noise to test if spectral features remain stable.
*   **Volume Variance**: Scales audio to ensure classification isn't dependent on loudness.
