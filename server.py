from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import io
import soundfile as sf
import time
import torch
import joblib

# Import our existing agents
from orchestrator import SentinelNeuralOrchestrator

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize Agents Global
print("🚀 Initializing Sentinel AI Engines (Multi-Model Orchestrator)...")
live_agent = SentinelNeuralOrchestrator(mode="live")
file_agent = SentinelNeuralOrchestrator(mode="file")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_live', methods=['POST'])
def detect_live():
    try:
        # Get audio blob from frontend
        audio_file = request.files['audio']
        
        # Read into numpy array
        # We use soundfile or librosa. load handles WAV best without ffmpeg
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            # Try loading with librosa (resample to 16k)
            audio, sr = librosa.load(audio_buffer, sr=16000)
        except Exception as le:
            print(f"Librosa load failed: {le}. Attempting raw soundfile read.")
            # Fallback for systems with missing codecs
            audio_buffer.seek(0)
            audio, sr = sf.read(audio_buffer)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Run Detection through the Orchestrator
        result = live_agent.analyze(audio)
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "Error", "message": str(e), "confidence": 0})

@app.route('/detect_file', methods=['POST'])
def detect_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        
        # Load Analysis (Handling potential codec issues)
        audio, sr = librosa.load(file, sr=16000)
        
        # Run Analysis through the Orchestrator
        # Reset memory to ensure this file is analyzed as a fresh forensic event
        file_agent.reset_memory()
        result = file_agent.analyze(audio)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e), "confidence": 0})

@app.route('/system_health', methods=['GET'])
def system_health():
    try:
        # Benchmarking Logic
        # 1.5 seconds * 16000 Hz = 24000 samples
        dummy_chunk = np.random.uniform(-1, 1, 24000).astype(np.float32)
        latencies = []
        for _ in range(10): # Run 10 times for speed
            t0 = time.time()
            live_agent.analyze(dummy_chunk)
            latencies.append(time.time() - t0)
            
        avg_ms = np.mean(latencies) * 1000
        throughput = 1 / (avg_ms / 1000)
        rt_factor = 1.5 / (avg_ms / 1000)
        
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        
        return jsonify({
            "latency_ms": round(avg_ms, 2),
            "throughput": round(throughput, 1),
            "rt_factor": round(rt_factor, 1),
            "device": device
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🌍 Sentinel Web Server starting at http://localhost:5000")
    app.run(debug=False, port=5000, host='0.0.0.0')
