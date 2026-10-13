import sounddevice as sd
samplerate = 16000
duration = 5
channels = 1
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
sd.wait()
sd.play(myrecording, samplerate)
sd.wait()


