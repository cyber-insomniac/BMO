import numpy as np, sounddevice as sd
a = sd.rec(16000 * 4, samplerate=16000, channels=1, dtype="int16"); sd.wait()
a = a[:, 0].astype(np.float32)
for i in range(0, len(a), 4000):
    w = a[i:i + 4000]
    print(f"mean {w.mean():7.0f}  std {w.std():6.0f}  peak {np.abs(w).max():6.0f}")
print("clipped samples:", int((np.abs(a) >= 32767).sum()))
