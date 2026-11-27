import numpy as np, sounddevice as sd
from livekit.wakeword import WakeWordModel
from pathlib import Path

BASE = Path(__file__).parent
ROOT = BASE.parent
WAKE_WORD_FILE      = "/home/insomnia/Projects/BMO/sound_models/hey_BMO.onnx"

sd.default.device = ("pipewire", "pipewire")
model = WakeWordModel(models=[str(WAKE_WORD_FILE)])    # use your real path
SR, BLOCK, WINDOW = 16000, 1280, 32000
GAIN = 1.0                                        # try 2.0 or 3.0 later
buf = np.zeros(WINDOW, dtype=np.int16)

with sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK, dtype="int16") as s:
    n = 0
    while True:
        chunk = s.read(BLOCK)[0][:, 0]
        buf = np.concatenate((buf[len(chunk):], chunk))
        n += 1
        if n % 2:
            continue
        x = chunk.astype(np.float32)
        x -= x.mean()
        level = float(np.sqrt(np.mean(x ** 2)))
        scored = np.clip(buf.astype(np.float32) * GAIN, -32768, 32767).astype(np.int16)
        score = model.predict(scored).get("hey_BMO", 0)
        print(f"level {level:6.0f}  peak {np.abs(chunk).max():6d}  wake {score:.2f}  {'#' * int(score * 30)}")
