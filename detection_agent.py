import torch
import joblib
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class DeepfakeGuardianAgent:
    def __init__(self):
        print("Initializing Agent...")
        model_path = "./model_store" 
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path)
        
        try:
            self.brain = joblib.load("agent_brain.pkl")
            print("🧠 Custom Agent Brain Loaded!")
        except:
            print("⚠️ WARNING: No brain found. Run train_agent.py first!")
            self.brain = None
            
        self.history = []

    def extract_features(self, audio_data):
        inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

    def detect(self, audio_chunk):
        # 1. Silence Check
        energy = np.sqrt(np.mean(audio_chunk**2))
        if energy < 0.005: 
             return {"status": "🔇 STANDBY", "confidence": 0.0, "reason": "Environment Silent"}

        # 2. Feature Extraction
        try:
            features = self.extract_features(audio_chunk)
            
            # 3. Prediction
            if self.brain:
                prob_fake = self.brain.predict_proba([features])[0][1] 
            else:
                prob_fake = 0.5

            # 4. Temporal Consensus (Optimized for 1.5s Chunks)
            self.history.append(prob_fake)
            if len(self.history) > 5: self.history.pop(0)
            avg_risk = sum(self.history) / len(self.history)

            # 5. Decision
            if len(self.history) < 3:
                 return {"status": "🔍 CALIBRATING", "confidence": avg_risk, "reason": "Analyzing Voice..."}
                 
            if avg_risk > 0.5:
                return {"status": "🚨 DEEPFAKE DETECTED", "confidence": avg_risk, "reason": "AI spectral signature found"}
            else:
                return {"status": "✅ VERIFIED HUMAN", "confidence": avg_risk, "reason": "Natural vocal signal confirmed"}
        except Exception as e:
            return {"status": "Error", "message": str(e), "confidence": 0}