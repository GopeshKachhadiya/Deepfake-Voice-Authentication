# train_file_agent.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import joblib
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from prosody_model import ProsodyLSTM, extract_prosody_features

# 1. LOAD LOCAL MODEL
print("Loading Local WavLM Industrial Model for Forensic Training...")
model_path = "./model_store"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = WavLMModel.from_pretrained(model_path)
model.eval()
print("Model Loaded.")

def extract_spectral_features(audio_data):
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

def augment_audio(audio_data):
    noise = np.random.normal(0, 0.005, len(audio_data))
    noisy_audio = audio_data + noise
    # Simpler volume variation for file augmentation
    varied_audio = audio_data * 0.5
    return [noisy_audio, varied_audio]

def process_and_augment(audio_data, X_spectral, X_prosody, y, label):
    X_spectral.append(extract_spectral_features(audio_data))
    X_prosody.append(extract_prosody_features(audio_data))
    y.append(label)
    for aug in augment_audio(audio_data):
        X_spectral.append(extract_spectral_features(aug))
        X_prosody.append(extract_prosody_features(aug))
        y.append(label)

# 2. DATA COLLECTION
X_spectral = []
X_prosody = []
y = []

print("\n--- FORENSIC MULTI-MODEL TRAINING (Pipeline + LSTM) ---")
mode = input("Train using (1) Microphone or (2) Existing Folders? [1/2]: ")

if mode == "1":
    import pyaudio
    def record_audio(seconds=4):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        print("🔴 Speak now!")
        frames = [np.frombuffer(stream.read(1024), dtype=np.float32) for _ in range(int(16000/1024*seconds))]
        stream.stop_stream(); stream.close(); p.terminate()
        return np.hstack(frames)

    for label_str, val in [("REAL", 0), ("FAKE", 1)]:
        for i in range(5):
            input(f"  [{label_str} {i+1}/5] Record...")
            process_and_augment(record_audio(), X_spectral, X_prosody, y, val)

elif mode == "2":
    real_path = input("Enter path to REAL folder: ").strip().replace('"', '')
    fake_path = input("Enter path to FAKE folder: ").strip().replace('"', '')
    for folder, label in [(real_path, 0), (fake_path, 1)]:
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if f.endswith('.wav')][:5]
        for f in files:
            audio, _ = librosa.load(os.path.join(folder, f), sr=16000)
            process_and_augment(audio, X_spectral, X_prosody, y, label)

# 3. TRAIN MODELS
if len(X_spectral) > 0:
    print("Training Spectral RF...")
    spectral_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))])
    spectral_pipeline.fit(X_spectral, y)
    joblib.dump(spectral_pipeline, "file_agent_brain.pkl")

    print("Training Prosody LSTM...")
    prosody_net = ProsodyLSTM()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(prosody_net.parameters(), lr=0.001)
    
    max_len = max([feat.shape[0] for feat in X_prosody])
    X_pro_padded = [np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant') for feat in X_prosody]
    X_pro_tensor = torch.tensor(np.array(X_pro_padded), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    prosody_net.train()
    for e in range(50):
        optimizer.zero_grad()
        loss = criterion(prosody_net(X_pro_tensor), y_tensor)
        loss.backward(); optimizer.step()
    
    torch.save(prosody_net.state_dict(), "file_prosody_brain.pth")
    # Save cache for potential active learning
    np.savez("file_training_data_multi.npz", X_spec=X_spectral, X_pro=X_pro_padded, y=y)
    print("💾 Forensic Multi-Model Brains Saved.")
