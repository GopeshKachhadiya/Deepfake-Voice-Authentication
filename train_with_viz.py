import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prosody_model import ProsodyLSTM, extract_prosody_features

# --- SETTINGS ---
print("\n--- DATA CONFIGURATION ---")
TRAIN_REAL = input("Enter path to TRAINING REAL folder: ").strip().replace('"', '')
TRAIN_FAKE = input("Enter path to TRAINING FAKE folder: ").strip().replace('"', '')
VAL_REAL = input("Enter path to VALIDATION REAL folder: ").strip().replace('"', '')
VAL_FAKE = input("Enter path to VALIDATION FAKE folder: ").strip().replace('"', '')

EPOCHS = 80
BATCH_SIZE = 4
LEARNING_RATE = 0.001

def augment_audio(audio_data):
    """Mirroring the production augmentation logic for data consistency"""
    noise = np.random.normal(0, 0.01, len(audio_data))
    noisy = audio_data + noise
    # Simple volume variations for viz script
    quieter = audio_data * 0.5
    louder = audio_data * 1.2
    return [noisy, quieter, louder]

def load_directory_data(real_path, fake_path, limit=None, augment=True):
    X_prosody = []
    y = []
    
    print(f"📂 Processing: {os.path.basename(real_path)} & {os.path.basename(fake_path)}")
    for folder, label in [(real_path, 0), (fake_path, 1)]:
        if not os.path.exists(folder):
            print(f"⚠️ Warning: Folder {folder} not found.")
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        if limit: files = files[:limit]
        
        for f in files:
            try:
                file_path = os.path.join(folder, f)
                audio, _ = librosa.load(file_path, sr=16000, duration=10) # Limit duration to 10s
                
                # Original
                X_prosody.append(extract_prosody_features(audio))
                y.append(label)
                
                # Augmentations (Only if enabled)
                if augment:
                    for aug in augment_audio(audio):
                        X_prosody.append(extract_prosody_features(aug))
                        y.append(label)
            except Exception as e:
                print(f"Skipping {f}: {e}")
    
    return X_prosody, y

def calculate_accuracy(y_pred, y_true):
    predictions = (y_pred > 0.5).float()
    correct = (predictions == y_true).float().sum()
    return correct / y_true.shape[0]

def train_and_visualize():
    # 1. Load Training Data
    print("\n--- LOADING TRAINING SET ---")
    X_train_raw, y_train_raw = load_directory_data(TRAIN_REAL, TRAIN_FAKE, limit=5, augment=True)
    
    # 2. Load Explicit Validation Data (Reduced limit to prevent Memory Error)
    print("\n--- LOADING VALIDATION SET ---")
    X_val_raw, y_val_raw = load_directory_data(VAL_REAL, VAL_FAKE, limit=10, augment=False) 

    # 3. Global Padding (Ensure all sequences match across both sets)
    all_seqs = X_train_raw + X_val_raw
    max_len = max([f.shape[0] for f in all_seqs])
    
    X_train_padded = [np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='constant') for f in X_train_raw]
    X_val_padded = [np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='constant') for f in X_val_raw]
    
    # Convert to Tensors
    X_train_t = torch.tensor(np.array(X_train_padded), dtype=torch.float32)
    y_train_t = torch.tensor(np.array(y_train_raw), dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(np.array(X_val_padded), dtype=torch.float32)
    y_val_t = torch.tensor(np.array(y_val_raw), dtype=torch.float32).unsqueeze(1)
    
    model = ProsodyLSTM(input_size=39) # Updated for enhanced feature extraction
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrics to track
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print(f"\n🚀 Multi-Source Training started for {EPOCHS} epochs...")
    print(f"Training on {len(X_train_raw)} samples (with aug), Validating on {len(X_val_raw)} samples.")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        train_outputs = model(X_train_t)
        loss = criterion(train_outputs, y_train_t)
        train_acc = calculate_accuracy(train_outputs, y_train_t)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation on the EXPLICIT set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_acc = calculate_accuracy(val_outputs, y_val_t)
            
        # Record results
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4f}, Validation Accuracy: {val_acc.item():.4f}")

    # --- PLOTTING ---
    plt.style.use('dark_background')
    epochs_range = range(1, EPOCHS + 1)
    
    # 1. Accuracy Graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', color='#00f2ff', linewidth=2)
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', color='#7000ff', linewidth=2)
    plt.title('Sentinel AI | Prosody Neural Accuracy', fontsize=14, color='#00f2ff')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.1)
    plt.savefig('accuracy_plot.png')
    print("\n✅ Accuracy graph saved as 'accuracy_plot.png'")
    
    # 2. Loss Graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#ff2a6d', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#ffcc00', linewidth=2)
    plt.title('Sentinel AI | Training Loss Convergence', fontsize=14, color='#ff2a6d')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend()
    plt.grid(alpha=0.1)
    plt.savefig('loss_plot.png')
    print("✅ Loss graph saved as 'loss_plot.png'")
    
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
