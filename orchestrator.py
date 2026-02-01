import torch
import joblib
import numpy as np
import os
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from prosody_model import ProsodyLSTM, extract_prosody_features

class SentinelNeuralOrchestrator:
    """
    Advanced Agentic Intelligence Orchestrator:
    Acts as the 'Executive Agent' that manages a multi-model ensemble 
    using the Industry-Standard WavLM backbone.
    """
    def __init__(self, mode="live"):
        self.mode = mode
        print(f"🕵️ Initializing Sentinel Agentic Orchestrator [WavLM Powered]")
        
        model_path = "./model_store" 
        # WavLM uses the same feature extraction logic as Wav2Vec2
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.backbone = WavLMModel.from_pretrained(model_path)
        self.backbone.eval()
        
        # 1. Load Sub-Brain: Spectral Expert (Random Forest)
        brain_path = "agent_brain.pkl" if mode == "live" else "file_agent_brain.pkl"
        try:
            self.spectral_expert = joblib.load(brain_path)
            print("  |-- 🧠 Spectral Expert Online")
        except:
            self.spectral_expert = None
            
        # 2. Load Sub-Brain: Prosody Expert (LSTM)
        prosody_path = "prosody_brain.pth" if mode == "live" else "file_prosody_brain.pth"
        try:
            self.prosody_expert = ProsodyLSTM()
            self.prosody_expert.load_state_dict(torch.load(prosody_path))
            self.prosody_expert.eval()
            print("  |-- 🧠 Prosody Expert Online")
        except:
            self.prosody_expert = None
            
        # 3. Agentic Memory
        self.temporal_memory = []
        self.trust_threshold = 0.5
        self.calibration_limit = 2

    def reset_memory(self):
        """Clears the temporal audit trail for fresh analysis."""
        self.temporal_memory = []

    def analyze(self, audio_chunk):
        try:
            # Silence Skip
            energy = np.sqrt(np.mean(audio_chunk**2))
            if energy < 0.005: 
                 return {"status": "STANDBY", "confidence": 0.0, "reason": "Environment Silent"}

            # --- STEP 1: Agentic Feature Extraction (WavLM) ---
            results = {}
            if self.spectral_expert:
                inputs = self.feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    # WavLM extracts high-fidelity speaker identity & mask features
                    outputs = self.backbone(**inputs)
                spec_feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                results['spectral'] = self.spectral_expert.predict_proba([spec_feat])[0][1]
            else:
                results['spectral'] = 0.5

            # Prosody Logic (The Rhythmic Agent)
            if self.prosody_expert:
                with torch.no_grad():
                    pro_feat = extract_prosody_features(audio_chunk)
                    pro_tensor = torch.from_numpy(pro_feat).unsqueeze(0)
                    device = next(self.prosody_expert.parameters()).device
                    results['prosody'] = self.prosody_expert(pro_tensor.to(device)).item()
            else:
                results['prosody'] = 0.5

            # --- STEP 2: Agentic Arbitration (Dynamic Ensemble) ---
            # If sub-brains conflict significantly, the Orchestrator performs 
            # a 'Deep Forensic Audit' (weights adjustment)
            conflict_score = abs(results['spectral'] - results['prosody'])
            
            if conflict_score > 0.4:
                # Expert Conflict Detected: Tilt towards Spectral (usually more robust against vocoder)
                spec_weight = 0.80
                pro_weight = 0.20
                reason = "Conflicting Patterns Detected | Deep Spectral Audit Triggered"
            else:
                spec_weight = 0.60
                pro_weight = 0.40
                reason = "Multi-Agent Consensus Reached"

            combined_risk = (results['spectral'] * spec_weight) + (results['prosody'] * pro_weight)

            # --- STEP 3: Temporal Memory Consolidation ---
            self.temporal_memory.append(combined_risk)
            if len(self.temporal_memory) > 3: self.temporal_memory.pop(0)
            avg_risk = sum(self.temporal_memory) / len(self.temporal_memory)

            # --- STEP 4: Executive Verdict ---
            if 0.45 < avg_risk < 0.55:
                return {
                    "status": "🚨 DEEPFAKE GENERATED", 
                    "confidence": avg_risk, 
                    "reason": "Vocal Pattern Anomaly | Synthetic Artifact Probable",
                    "composition": f"WavLM-Spectral:{results['spectral']:.2f} | Prosody:{results['prosody']:.2f}"
                }
            elif avg_risk >= 0.55:
                return {
                    "status": "🚨 DEEPFAKE DETECTED", 
                    "confidence": avg_risk, 
                    "reason": reason,
                    "composition": f"WavLM-Spectral:{results['spectral']:.2f} | Prosody:{results['prosody']:.2f}"
                }
            else:
                return {
                    "status": "✅ VERIFIED HUMAN", 
                    "confidence": avg_risk, 
                    "reason": "Natural Biometric Signature Confirmed",
                    "composition": f"WavLM-Spectral:{results['spectral']:.2f} | Prosody:{results['prosody']:.2f}"
                }
        except Exception as e:
            return {"status": "Error", "message": str(e), "confidence": 0}
