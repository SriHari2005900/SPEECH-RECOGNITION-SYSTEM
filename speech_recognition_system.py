# Step 1: Install dependencies
!pip install --quiet SpeechRecognition pydub transformers torchaudio soundfile librosa

# Step 2: Import libraries
import speech_recognition as sr
from pydub import AudioSegment
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa, soundfile as sf
import numpy as np
import os

# -----------------------------------------------------------
# Helper Function to convert audio to WAV 16kHz mono
# -----------------------------------------------------------
def convert_to_wav_16k_mono(input_path, output_path="converted_audio.wav"):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path

# -----------------------------------------------------------
# Method 1: Using SpeechRecognition Library (Google)
# -----------------------------------------------------------
def transcribe_with_speechrecognition(audio_path):
    recognizer = sr.Recognizer()
    wav_file = convert_to_wav_16k_mono(audio_path)
    with sr.AudioFile(wav_file) as source:
        print("Listening to audio...")
        audio_data = recognizer.record(source)
    print("Transcribing using Google SpeechRecognition...")
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Audio not clear or speech not understood."
    except sr.RequestError:
        return "Error connecting to the recognition service."

# -----------------------------------------------------------
# Method 2: Using Pre-Trained Wav2Vec2 Model (Offline)
# -----------------------------------------------------------
def transcribe_with_wav2vec2(audio_path):
    print("Loading Wav2Vec2 pre-trained model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    print("Reading and processing audio...")
    wav_file = convert_to_wav_16k_mono(audio_path)
    speech, sr = librosa.load(wav_file, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    print("Running model for transcription...")
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# -----------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------
print("=======================================")
print("      SPEECH RECOGNITION SYSTEM        ")
print("=======================================")
print("Upload or provide an audio file path (.wav or .mp3)")
print("Example: sample.wav or test_audio.mp3")

# For Google Colab users: you can upload file manually
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    audio_path = filename
    print(f"\nAudio file uploaded: {audio_path}")

    # 1️⃣ Using SpeechRecognition
    print("\n--- Using SpeechRecognition (Google) ---")
    text_google = transcribe_with_speechrecognition(audio_path)
    print("Recognized Text (Google):")
    print(text_google)

    # 2️⃣ Using Wav2Vec2
    print("\n--- Using Wav2Vec2 (Hugging Face) ---")
    text_wav2vec = transcribe_with_wav2vec2(audio_path)
    print("Recognized Text (Wav2Vec2):")
    print(text_wav2vec)

print("\n====================")
print(" Transcription Done ")
print("====================")
