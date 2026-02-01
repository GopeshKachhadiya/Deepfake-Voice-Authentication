from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import os

# 1. Define the model ID - WavLM is superior for speaker-related tasks
model_name = "microsoft/wavlm-base-plus"

print(f"⏳ Downloading Industry-Standard: {model_name}...")

# 2. Download Model and Feature Extractor
# We use FeatureExtractor because we don't need the text vocabulary (Processor) for forensics
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name)

# 3. Create model_store if it doesn't exist
if not os.path.exists("./model_store"):
    os.makedirs("./model_store")

# 4. Save them to a local folder named 'model_store'
feature_extractor.save_pretrained("./model_store")
model.save_pretrained("./model_store")

print("✅ Industry-Standard WavLM model saved to './model_store'.")
print("🔥 This model provides better robustness against noise and speaker identity verification.")