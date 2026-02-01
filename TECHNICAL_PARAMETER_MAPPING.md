# System Architecture & Technical Mapping

This document provides a technical blueprint of the **Sentinel Neural Orchestrator**, detailing how it converts raw audio into a forensic verdict.

---

## 1. End-To-End Architecture Overview

### A. Data Input Stage
*   **Sources**: Live microphone stream (via PyAudio) or File-based input (`.wav`).
*   **Specs**: 16,000 Hz Sampling Rate, Mono-channel.
*   **Preprocessing**: 
    *   **Rolling Buffer**: Audio is processed in overlapping 1.5 - 4.0 second "time windows."
    *   **Energy Gate**: A silence check ensures the system ignores background noise (Energy < 0.005).

### B. Dual-Branch Feature Extraction (The "Brain")
The orchestrator splits the signal into two parallel analytical pipelines:

1.  **The Acoustic Branch (A.I. Artifact Detection)**
    *   **Framework**: `Transformers` (Hugging Face), `Scikit-learn`.
    *   **Model**: **Wav2Vec2** (Feature Extractor) + **Random Forest** (Classifier).
    *   **Logic**: Captures high-dimensional "latent" signatures of synthetic speech (vocoder scars, phase shifts).

2.  **The Rhythmic Branch (Biometric Analysis)**
    *   **Framework**: `PyTorch`, `Librosa`.
    *   **Model**: **ProsodyLSTM** (Bidirectional LSTM).
    *   **Logic**: Analyzes a sequence of **13 MFCCs** to map the "biological rhythm" and vocal tract patterns over time.

### C. Agentic Arbitration & Decision
*   **Weighted Fusion**: 
    *   `Final_Score = (Spectral_Score * 0.65) + (Prosody_Score * 0.35)`
*   **Critical Override**: If the Spectral branch detects a signature > 85%, it triggers an automatic "Deepfake" alert regardless of the rhythm score.
*   **Temporal Memory**: The last 3-5 decisions are averaged to prevent "flickering" and ensure forensic stability.

### D. System Output
*   **Classification**: `VERIFIED HUMAN` vs `🚨 DEEPFAKE DETECTED`.
*   **Confidence**: Probability value (0.0 to 1.0).
*   **Reasoning**: Natural Language explanation (e.g., "Critical Vocoder Artifact" or "Natural Biometric Rhythm").
*   **Composition**: Real-time ratio showing the score contribution from both experts.

---

## 2. Technical Frameworks Used
| Tool | Purpose |
| :--- | :--- |
| **PyTorch** | Deep Learning engine for the LSTM model. |
| **Hugging Face** | Hosting the pre-trained Wav2Vec2 transformer logic. |
| **Librosa** | Extraction of MFCCs and signal processing. |
| **Scikit-Learn** | Random Forest classification and feature scaling pipelines. |
| **Joblib** | Serializing and loading the trained "Brain" models. |

---

## 3. Parameter Mapping
*These are directly calculated from the raw audio signal.*

| Parameter | System Component | Technical Implementation |
| :--- | :--- | :--- |
| **MFCCs (Mel-Frequency Cepstral Coefficients)** | Prosody Expert (LSTM) | Extracted via `librosa.feature.mfcc` (13 coefficients). Used to map the "vocal tract" signature. |
| **Loudness / Energy** | Signal Gate / Orchestrator | Calculated via Root Mean Square (RMS). Used for environment calibration and silence detection. |

---

## 2. Implicitly Analyzed Parameters (AI-Learned)
*Our Deep Learning models (Wav2Vec2 and LSTM) are trained to identify these patterns within the spectral and temporal data, even though they aren't "manually" calculated as single numbers.*

### A. Temporal & Rhythmic Patterns (Handled by LSTM)
These are captured through the sequence analysis of MFCCs over a 1.5 - 4.0 second window:
*   **Intonation / Prosody**: The LSTM tracks the "flow" of speech to see if natural rise-fall patterns exist.
*   **Speech Rate**: The model detects if the speed of phonemes is naturally variable or unnaturally constant.
*   **Pause Patterns**: Detects if hesitations and breaths occur in biologically likely locations.
*   **Micro-Prosody**: Captured via the frame-by-frame temporal transitions in the LSTM.

### B. Spectral & Synthetic Signatures (Handled by Wav2Vec2)
The "Spectral Expert" uses high-dimension embeddings to find these "smoking guns":
*   **Model Fingerprints**: Detects the specific mathematical "scars" left by vocoders like HiFi-GAN or WaveNet.
*   **Phase Coherence**: Pre-trained transformers are highly sensitive to phase shifts typical of neural synthesis.
*   **Spectral Artifacts**: Identifies "checkerboard" patterns or digital noise in high frequencies that humans can't hear.
*   **Cross-Frame Consistency**: Detects if the audio is "too stable" across time (a common flaw in AI generation).

---

## 3. Currently Unused Parameters
*These are not explicitly measured in the current version of the code:*
*   **Pitch (F0)**: Not tracked as a standalone frequency curve.
*   **Formants (F1–F4)**: Not explicitly isolated (though partially represented in MFCCs).
*   **Jitter & Shimmer**: Requires micro-stability analysis not currently in the pipeline.
*   **Harmonic-to-Noise Ratio (HNR)**: Not explicitly calculated.
*   **Voice Onset Time (VOT)**: Requires phoneme-level segmentation.

---

## Summary for Jury Pitch
"Our system focuses on **MFCCs** to understand the physical shape of the speaker's vocal tract and **Wav2Vec2 Embeddings** to find the 'Model Fingerprints' left by AI software. While a human hears a voice, our AI hears a mathematical pattern of **energy, rhythm, and spectral consistency**."
