import numpy as np
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import vosk
import wave
import logging

# Set logging to hide Vosk API warnings
logging.getLogger("VoskAPI").setLevel(logging.WARNING)

# Fix for deprecated numpy complex type
if not hasattr(np, 'complex'):  # Compatibility check
    np.complex = np.complex128

# Load the audio file
audio_path = 'd:/Users/HOME/Downloads/Music/cmvol4bk2_01_mason_64kb-[AudioTrimmer.com].mp3'
y, sr_rate = librosa.load(audio_path, sr=None)  # Load the audio file with its original sampling rate

# Step 1: Add noise to the audio file
np.random.seed(0)  # Set random seed for noise generation
noise = np.random.normal(0, 0.02, y.shape)  # Generate Gaussian noise
y_noisy = y + noise  # Add noise to the original signal

# Save noisy audio
noisy_audio_path = "C:/Users/HOME/noisy_audio.wav"
sf.write(noisy_audio_path, y_noisy, sr_rate)

# Step 2: Denoise the audio
y_denoised = nr.reduce_noise(y=y_noisy, sr=sr_rate, stationary=True)  # Use noisereduce library to remove noise

# Save denoised audio
denoised_audio_path = "C:/Users/HOME/denoised_audio.wav"
sf.write(denoised_audio_path, y_denoised, sr_rate)

# Step 3: Convert audio to text using Vosk (offline)

# Convert the denoised WAV file to a format compatible with Vosk
sound = AudioSegment.from_wav(denoised_audio_path)  # Load the audio file with pydub
sound = sound.set_channels(1)  # Ensure the audio is mono
sound = sound.set_frame_rate(16000)  # Ensure the sample rate is 16 kHz
wav_output_path = "C:/Users/HOME/final_audio.wav"
sound.export(wav_output_path, format="wav")  # Export the audio file as WAV format

# Load the Vosk model
model_path = "D:/vosk-model-small-en-us-0.15"
from vosk import Model, KaldiRecognizer

try:
    model = Model(model_path)  # Load the Vosk model
    wf = wave.open(wav_output_path, "rb")  # Open the audio file
    # Check the audio file format for compatibility
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        raise ValueError("Audio file must be WAV format mono PCM with a sample rate of 8000 or 16000 Hz.")

    recognizer = KaldiRecognizer(model, wf.getframerate())  # Create the Recognizer object for speech recognition
    recognizer.SetWords(True)  # Enable word-level recognition

    text = ""  # Variable to store the recognized text
    while True:
        data = wf.readframes(4000)  # Read data from the audio file
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):  # Check if the data has been fully processed
            result = recognizer.Result()  # Get the recognition result
            result_dict = eval(result)  # Convert the result to a dictionary
            if 'text' in result_dict:
                text += result_dict['text'] + " "  # Append recognized text to the variable

    # Get the final recognition result
    final_result = recognizer.FinalResult()  # Get the final result
    final_result_dict = eval(final_result)  # Convert it to a dictionary
    if 'text' in final_result_dict:
        text += final_result_dict['text']  # Append the final recognized text

    # Print all recognized text
    print(text)

except Exception as e:
    print("Error during speech recognition:", str(e))  # If an error occurs, print the error message
