import sounddevice as sd
samplerate = 16000
duration = 5
channels = 1
print("Start")
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
sd.wait()
print("play")
sd.play(myrecording, samplerate)
sd.wait()


