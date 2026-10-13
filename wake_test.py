import openwakeword
from openwakeword.model import Model
import numpy as np
import sounddevice as sd

openwakeword.utils.download_models()

model = Model(
  wakeword_models=["hey_jarvis"],
)

SAMPLERATE = 16000
CHANNELS = 1

duration = 5

def detect_wake_word(audio):

    prediction = model.predict(audio)
    score = prediction.get("hey_jarvis", 0)

    return score > 0.5

def callback(indata, frames, time, status):
    
    if status:
        print(status)

    audio_chunk = indata[:, 0].copy()

    if detect_wake_word(audio_chunk):
        print("WOKE")

with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=1280, dtype="int16", callback=callback):
    sd.sleep(int(duration * 1000))

