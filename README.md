# 🛡️ Sentinel AI — Real-Time Deepfake Voice Authentication Bypass Detection

> **Protecting voice biometric authentication in live VoIP calls** — multi-model AI pipeline fusing spectral analysis, temporal LSTM, and agentic orchestration to detect AI-generated voice clones in real time.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)

---

## 🔍 Overview

**Sentinel AI** is a real-time deepfake voice detection system engineered to prevent voice biometric authentication bypass in live VoIP and phone-based authentication systems. As AI voice cloning tools (ElevenLabs, XTTS, Tortoise-TTS, etc.) become increasingly accessible, traditional voice authentication systems are critically vulnerable to spoofing attacks.

Sentinel AI combats this by fusing multiple specialized AI models — a Transformer-based spectral scanner, an LSTM temporal analyzer, and an agentic fusion layer — to produce a real-time authenticity verdict within a 4-second analysis window, making it suitable for live call environments.

---

## ✨ Features

- ⚡ **Real-Time Detection** — 4.0s analysis window optimized for live VoIP streams
- 🎙️ **Multi-Expert Fusion** — Ensemble of Transformer spectral + LSTM temporal models
- 🕵️ **Agentic Orchestration** — AI agents coordinate model outputs for a final verdict
- 📞 **Telephony-Robust** — Handles network noise, codec compression, and telephony filters
- 🧠 **Short-Audio Optimized** — Works accurately on 3–12 second voice samples
- 🎭 **Context-Aware** — Distinguishes deepfakes from natural voice variation (stress, illness, emotion)
- 🌐 **Web Interface** — Browser-based real-time audio capture and analysis via Flask + JavaScript
- 📊 **Confidence Scoring** — Returns probability score and per-model breakdown

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | [Python](https://python.org/), [Flask](https://flask.palletsprojects.com/) |
| **Frontend** | JavaScript (Web Audio API for real-time audio capture) |
| **Spectral Model** | Transformers (HuggingFace) — frequency & spectral artifact detection |
| **Temporal Model** | LSTM (PyTorch/TensorFlow) — temporal pattern analysis |
| **Agent Layer** | Custom agentic orchestrator for multi-model fusion |
| **Audio Processing** | librosa, scipy, numpy |
| **Language** | Python 3.10+, JavaScript (ES6) |

---

## 🏗️ Architecture

```
Deepfake-Voice-Authentication/
├── server.py                   # Flask backend entry point
├── orchestrator.py             # Agentic fusion layer
├── prosody_model.py            # Temporal analyzer
├── detection_agent.py          # Real-time inference agent
├── file_detection_agent.py     # File inference agent
├── templates/
│   └── index.html              # Web UI
├── static/                     # External scripts/styles
├── requirements.txt
└── .env
```

---

## ⚙️ How It Works

```
Live Audio Input (VoIP / Mic)
        │
        ▼
┌─────────────────────┐
│  Audio Preprocessor │  → Normalize, denoise, extract MFCC/mel-spectrograms
└─────────────────────┘
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
┌──────────────────┐                    ┌──────────────────────┐
│  Spectral Model  │                    │   Temporal Model     │
│  (Transformer)   │                    │   (LSTM)             │
│  Frequency &     │                    │   Timing & rhythm    │
│  artifact scan   │                    │   pattern analysis   │
└──────────────────┘                    └──────────────────────┘
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
            ┌─────────────────────┐
            │  Agentic Fusion     │  → Weighted ensemble decision
            │  Orchestrator       │
            └─────────────────────┘
                       │
                       ▼
          ✅ AUTHENTIC  /  ❌ DEEPFAKE
          + Confidence Score (0–100%)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip / conda
- Modern web browser (for UI)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/GopeshKachhadiya/Deepfake-Voice-Authentication.git
cd Deepfake-Voice-Authentication

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Add your model API keys or local model paths

# 5. Run the Flask server
python server.py
```

---

## 🚀 Usage

1. Open `http://localhost:5000` in your browser
2. Click **"Start Recording"** — the system captures audio via your microphone
3. Speak for 3–12 seconds
4. Click **"Analyze"** — results appear in under 4 seconds
5. View:
   - ✅ **Authentic** or ❌ **Deepfake** verdict
   - Confidence score (%)
   - Per-model breakdown (Spectral / Temporal scores)

### API Usage

```bash
# POST audio file for analysis
curl -X POST http://localhost:5000/detect_file \
  -F "file=@sample.wav" \
  -H "Content-Type: multipart/form-data"
```

**Response:**
```json
{
  "verdict": "DEEPFAKE",
  "confidence": 94.7,
  "spectral_score": 0.93,
  "temporal_score": 0.96,
  "processing_time_ms": 812
}
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/detect_live` | POST | Analyze live audio stream |
| `/detect_file` | POST | Analyze uploaded audio file |
| `/system_health` | GET | Server health check benchmark |

---

## ⚠️ Disclaimer

This tool is designed for **defensive security research** and legitimate enterprise authentication hardening. Do not use for any malicious or unauthorized surveillance purposes.

---

<div align="center">Built with ❤️ to protect voice identity in the age of AI</div>
