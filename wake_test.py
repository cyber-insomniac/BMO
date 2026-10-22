'''
from openwakeword.model import Model

openwakeword.utils.download_models()

model = Model(
  wakeword_models=["hey_jarvis"],
)
'''

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel

MODEL_FILE = "hey_BMO.onnx"   # must match the real file name exactly
KEY = "hey_BMO"               # file name without .onnx, same capitalization
THRESHOLD = 0.5
SAMPLERATE = 16000
BLOCKSIZE = 1280
WINDOW = 2 * SAMPLERATE       # keep the last 2 seconds of audio

model = WakeWordModel(models=[MODEL_FILE])
buffer = np.zeros(WINDOW, dtype=np.int16)

def callback(indata, frames, time, status):
    global buffer
    if status:
        print(status)

    chunk = indata[:, 0]
    buffer = np.concatenate((buffer[len(chunk):], chunk))  # slide the window

    scores = model.predict(buffer)
    print(scores)                       # debug: remove once it works
    if scores.get(KEY, 0) > THRESHOLD:
        print("WOKE")

with sd.InputStream(samplerate=SAMPLERATE, channels=1, blocksize=BLOCKSIZE,
                    dtype="int16", callback=callback):
    sd.sleep(30_000)   # listen for 30 seconds
