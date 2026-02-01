import torch
import torch.nn as nn
import librosa
import numpy as np

class ProsodyLSTM(nn.Module):
    def __init__(self, input_size=39, hidden_size=64, num_layers=2): # 13 MFCC + 13 Delta + 13 Delta-Delta
        super(ProsodyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def extract_prosody_features(audio_data, sr=16000):
    """
    Enhanced extraction: MFCCs + Deltas + Delta-Deltas.
    This captures micro-prosody and vocal tract transition dynamics.
    """
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack features: (39, frames)
    combined = np.vstack([mfcc, delta, delta2])
    return combined.T.astype(np.float32)

def get_prosody_score(model, audio_data):
    model.eval()
    with torch.no_grad():
        features = extract_prosody_features(audio_data)
        # Add batch dimension
        features_tensor = torch.from_numpy(features).unsqueeze(0)
        # Handle sequence lengths if necessary, but here we just pass it
        prediction = model(features_tensor)
        return prediction.item()
