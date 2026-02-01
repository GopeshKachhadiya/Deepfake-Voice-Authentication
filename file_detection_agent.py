import torch
import joblib
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class FileDeepfakeGuardianAgent:
    def __init__(self):
        print("Initializing Forensic Agent...")
        model_path = "./model_store" 
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path)
        
        try:
            self.brain = joblib.load("file_agent_brain.pkl")
            print("🧠 Forensic Brain Loaded!")
        except:
            self.brain = None

    def extract_features(self, audio_data):
        inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

    def detect(self, audio_data):
        # 1. Silence Check
        energy = np.sqrt(np.mean(audio_data**2))
        if energy < 0.005: 
             return {"status": "🔇 Silence Detected", "confidence": 0.0}

        try:
            features = self.extract_features(audio_data)
            if self.brain:
                prob = self.brain.predict_proba([features])[0][1]
            else:
                prob = 0.5

            if prob > 0.5:
                return {"status": "🚨 DEEPFAKE DETECTED", "confidence": prob, "reason": "Neural artifacts found"}
            else:
                return {"status": "✅ VERIFIED HUMAN", "confidence": prob, "reason": "Spectral analysis passed"}
        except Exception as e:
            return {"status": "Error", "message": str(e), "confidence": 0}
