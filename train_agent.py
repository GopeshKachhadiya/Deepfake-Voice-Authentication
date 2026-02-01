# train_agent.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pyaudio
import numpy as np
import joblib
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import butter, lfilter

# Import our new Prosody Model
from prosody_model import ProsodyLSTM, extract_prosody_features

# 1. LOAD LOCAL MODEL
print("Loading Local WavLM Industrial Model...")
model_path = "./model_store"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = WavLMModel.from_pretrained(model_path)
model.eval()
print("Model Loaded.")

def record_audio(seconds=4, label="sample"):
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"\n🔴 RECORDING {label.upper()}... Speak now!")
    frames = []
    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    print("✅ Done.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.hstack(frames)

def extract_spectral_features(audio_data):
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

# --- ROBUSTNESS AUGMENTATIONS ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_telephony_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def apply_packet_loss(audio, loss_rate=0.05, chunk_size=160):
    """Simulates VoIP packet loss by dropping small chunks of audio."""
    processed = audio.copy()
    num_chunks = len(audio) // chunk_size
    for i in range(num_chunks):
        if np.random.rand() < loss_rate:
            processed[i*chunk_size : (i+1)*chunk_size] = 0
    return processed

def apply_bit_reduction(audio, bits=8):
    """Simulates low bit-depth quantization artifacts."""
    q = 2**(bits-1)
    return np.round(audio * q) / q

def augment_audio(audio_data):
    noise = np.random.normal(0, 0.01, len(audio_data))
    noisy_audio = audio_data + noise
    telephony_audio = apply_telephony_filter(audio_data)
    packet_loss_audio = apply_packet_loss(audio_data)
    quantized_audio = apply_bit_reduction(audio_data, bits=4) # Extreme 4-bit test
    varied_audio = audio_data * 0.4
    return [noisy_audio, telephony_audio, packet_loss_audio, quantized_audio, varied_audio]

def process_and_augment(audio_data, X_spectral, X_prosody, y, label):
    # Original
    X_spectral.append(extract_spectral_features(audio_data))
    X_prosody.append(extract_prosody_features(audio_data))
    y.append(label)
    
    # Augmented versions
    for aug in augment_audio(audio_data):
        X_spectral.append(extract_spectral_features(aug))
        X_prosody.append(extract_prosody_features(aug))
        y.append(label)

# 2. DATA COLLECTION
X_spectral = []
X_prosody = []
y = []

print("\n--- MULTI-MODEL AGENT TRAINING (Wav2Vec2 + Prosody LSTM) ---")
mode = input("Train using (1) Live Recording or (2) Folder of .wav files? [1/2]: ")

if mode == "1":
    for label_str, val in [("REAL", 0), ("FAKE", 1)]:
        print(f"\n--- PHASE: TEACHING {label_str} VOICE ---")
        num_samples = 5
        for i in range(num_samples):
            input(f"  [{label_str} Sample {i+1}/{num_samples}] Press Enter to Record 4 seconds...")
            audio = record_audio(seconds=4)
            process_and_augment(audio, X_spectral, X_prosody, y, val)

elif mode == "2":
    real_path = input("Enter path to REAL .wav files folder: ").strip().replace('"', '')
    fake_path = input("Enter path to FAKE .wav files folder: ").strip().replace('"', '')
    for folder, label in [(real_path, 0), (fake_path, 1)]:
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if f.endswith('.wav')][:5]
        for f in files:
            audio, _ = librosa.load(os.path.join(folder, f), sr=16000)
            process_and_augment(audio, X_spectral, X_prosody, y, label)

# 3. TRAIN BRANCH 1: SPECTRAL (Random Forest)
if len(X_spectral) > 0:
    print(f"\nTraining Spectral Brain (Random Forest)...")
    spectral_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    spectral_pipeline.fit(X_spectral, y)
    joblib.dump(spectral_pipeline, "agent_brain.pkl")

    # 4. TRAIN BRANCH 2: PROSODY (LSTM)
    print("Training Prosody Brain (LSTM)...")
    prosody_net = ProsodyLSTM()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(prosody_net.parameters(), lr=0.001)
    
    # Convert prosody data to tensors
    # Since sequences might vary slightly in frame count from librosa, we pad them
    max_len = max([feat.shape[0] for feat in X_prosody])
    X_pro_padded = []
    for feat in X_prosody:
        padded = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant')
        X_pro_padded.append(padded)
    
    X_pro_tensor = torch.tensor(np.array(X_pro_padded), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Simple training loop
    prosody_net.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = prosody_net(X_pro_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
    
    torch.save(prosody_net.state_dict(), "prosody_brain.pth")
    print("💾 Full Multi-Model Brain Saved (agent_brain.pkl + prosody_brain.pth).")
else:
    print("No data collected.")