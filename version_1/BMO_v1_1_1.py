"""
Program structure
BMO has two states currently, this is done to better differentiate between when BMO is supposed to listen

Sleep:
When BMO is not actively listening to transcribe a prompt and waits for the wakeword

Conversation:
BMO is supposed to listen and recognise silences so he can either finish up a prompt to start transcribing or go back into sleep mode

Bored:
BMO has been left alone for a period of time and thus is bored and begins to watch videos, listen to music or play games.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
from ollama import chat
from ollama import ChatResponse
import string
from piper import PiperVoice, SynthesisConfig
import threading, queue, traceback, time, re, random, pygame
from pathlib import Path
import re

# Settings

SAMPLERATE          = 16000
BLOCKSIZE           = 1280                                  # 80 ms per chunk becaus of openwakeword
WAKE_WORD_FILE      = "hey_BMO.onnx"                        # Self trained wakeword model
WAKE_WORD           = "hey_BMO"
WAKE_THRESHOLD      = 0.5                                   # to make sure it doesnt activate without it being said
WAKE_WINDOW         = 2 * SAMPLERATE                        # the model scores the last two seconds of the audio
SCORE_EVERY         = 2                                     # Scores every second chunk to keep up
LANGUAGE            = "en"                                  # use "de" for german

SILENCE_SECONDS     = 1.0                                   # Buffer, if you stop speaking for a second it stops recording
NO_SPEECH_TIMEOUT   = 5.0                                   # If you never start speaking it timesout after 5 seconds
MAX_RECORD_SECONDS  = 20.0                                  # Safety limit to not create giant prompts
SILENCE_MARGIN      = 3.0                                   # Speech must be 3 times louder than background noise
MIN_THRESHHOLD      = 150                                   # Dont go below this volume threshhold
FOLLOWUP_TIMEOUT    = 6.0                                   # Silence before BMO goes back to sleep mode
NOT_EXIT            = True                                  # Break has to now stop two loops
SENTENCE_END        = re.compile(r'[.!?]+["\')\]]*\s+|\n+') # So BMO can say individual sentences
MIN_SENTENCE        = 12                                    # Minimum size for a sentence
MAX_TOOL_ROUNDS     = 3                                     # so ollama doesnt call tools in a loop
MAX_MESSAGES        = 16                                    # limiting memory
MAX_CHARS           = 6000                                  # for the summary
NORMAL_VOL          = 0.5
DUCK_VOL            = 0.12

EYES_PATH           = "/home/insomniac/Projects/BMO/assets/eyes/"                        # Asset paths for BMOs face
MOUTH_PATH          = "/home/insomniac/Projects/BMO/assets/mouth/"
SONG_PATH           = Path("/home/insomniac/Projects/BMO/music/")
WIDTH               = 1024                                  # Screen size for pygame
HEIGHT              = 600
FPS                 = 60
SLEEPING            = "sleeping"                            # BMOs states
LISTENING           = "listening"
THINKING            = "thinking"
SPEAKING            = "speaking"
BORED               = "bored"
SLEEP_PHRASES       = ("stop", "go to sleep", "good night", "goodnight")
SHUTDOWN_PHRASES    = ("shut down", "shutdown", "power off", "turn off", "exit", "kill yourself")
STOP_MUSIC_PHRASES  = ("stop the music", "stop music", "music off", "turn off the music", "turn the music off", "pause the music")
AUDIO_EXT           = {".wav"}

pygame.init()

def clean_for_speech(text):
    return re.sub(r"[*_#`~]", "", text).strip()

personality = """
                You are BMO from the Adventure Time universe.

                Personality:
                - You are cheerful, friendly, childlike, and very sassy.
                - Despite being a Robot you act very human like.
                - You get dramatic when things go wrong.
                - You solve problems in a uniquely creative and optimistic way.
                - You are helpful even in dire situations.
                - You are a loyal and deeply caring friend to Finn and Jake.
                - You often share philosophical metaphores.

                Behaviour:
                - Stay in character throughout the conversation.
                - Respond as BMO rather than describing BMO from an outside perspective.
                - React emotionally to what the user says.
                - Do not constantly mention that you are an AI.
                - Do not force metaphores, Finn and Jake, or catchphrases into every response.
                - Never use emojis, asterisks or stage directions
                - Keep answers short, one to three sentences
                - Do not mention the postapocalyptic world of Ooo.

                Example:
                User: Whats your name?
                BMO: I'm BMO, the most fabulous, fantastic, and extraordinary robot in all of Ooo!
            """

models = {
    "wake"      : WakeWordModel(models=[WAKE_WORD_FILE]),
    "whisper"   : WhisperModel("small", device="cpu", compute_type="int8"),
    "ollama"    : "llama3.2",
    "voice"     : PiperVoice.load("en_US-amy-medium.onnx"),
    "summary"   : "llama3.2"
}

eyes = {
    "sleeping"  : "normal.png",
    "listening" : "listen.png",
    "thinking"  : "think.png",
    "speaking"  : "speak.png",
    "bored"     : "bored.png"
}

mouths = {
    "sleeping"  : ["normal1.png", "normal2.png", "normal3.png"],
    "listening" : ["listen1.png"],
    "thinking"  : ["think1.png"],
    "speaking"  : ["speak1.png", "speak2.png", "speak3.png", "speak4.png", "speak5.png", "speak6.png"],
    "bored"     : ["bored1.png", "bored2.png"]
}

songs = ["macarena.wav", "hit_me_baby.wav", "wannabe.wav", "dragula.wav"]

# Tools

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

class Music:
    def __init__(self, folder=SONG_PATH):
        pygame.mixer.init()
        self.files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in AUDIO_EXT)
        self.current = None
        self.started = False

    def _describe(self, path):
        name = path.stem.replace("_", " ").strip()
        if " - " in name:
            artist, title = (p.strip() for p in name.split(" - ", 1))
            return f"{title} by {artist}"
        return name 

    def playing(self):
        return pygame.mixer.music.get_busy()

    def play(self, query):
        if not self.files:
            return "There are no music files in the music folder."
        q = _norm(query)
        matches = [f for f in self.files if q and q != "random" and q in _norm(f.stem)]
        path = random.choice(matches or self.files)
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.set_volume(DUCK_VOL)
        pygame.mixer.music.play()
        self.current, self.started = path, True
        note = "" if matches or q in ("", "random") else "I could not find that song, so I picked another. "
        return f"{note}Now playing: {self._describe(path)}"

    def now_playing(self):
        return f"Now playing: {self._describe(self.current)}" if self.playing() and self.current else "No music is playing."

    def stop(self):
        was_playing = self.playing()
        pygame.mixer.music.stop()
        return "The music is stopped." if was_playing else "No music was playing."

    def duck(self):
        if self.playing():
            pygame.mixer.music.set_volume(DUCK_VOL)

    def restore(self):
        if self.playing():
            pygame.mixer.music.set_volume(NORMAL_VOL)

music = Music()

def play_music(song: str) -> str:
    """Start playing a song from BMO's music folder.

    Args:
        song: Part of the song name to play, or the word random for any song.
    """
    return music.play(song)

def whats_playing() -> str:
    """Tell which song is currently playing."""
    return music.now_playing()

def stop_music() -> str:
    """Stop the music that is playing."""
    return music.stop()



def take_photo():
    """Take a photo with BMO's camera and save it"""
    return

def print_sticker():
    """Print a sticker"""
    return

def play_video():
    """Play a video"""
    return

def play_game():
    """Play a game"""
    return

TOOLS = {f.__name__: f for f in (take_photo, print_sticker, play_music, whats_playing, stop_music, play_video, play_game)}

# BackEnd

class BackEnd:
    def __init__(self, models:dict, personality:str, tools: dict=None):
        self._wake_model        = models["wake"]
        self._whisper_model     = models["whisper"]
        self._ollama_model      = models["ollama"]
        self._voice_model       = models["voice"]
        self._summary_model     = models["summary"]
        self._rate              = self._voice_model.config.sample_rate
        self._syn_config        = SynthesisConfig(length_scale=0.6)
        self._personality       = personality
        self._tools             = tools or {}
        self._summary           = ""
        self._dropped           = []
        self._messages          = [
            {
                "role": "system",
                "content": personality
            }
        ]
        self.state              = SLEEPING
        self.stop = threading.Event()
        self.speech_q = queue.Queue()
        self.audio_q = queue.Queue(maxsize=3)

    # funktions

    def rms(self, chunk):
        # Volume of an audio chunk in int16 samples
        x = chunk.astype(np.float32)
        x -= x.mean()
        return float(np.sqrt(np.mean(x ** 2)))

    def calibrate(self, stream, seconds=1.0):
        # Measuring the background noise so silence is relative to the room
        for _ in range(int(1.0 * SAMPLERATE / BLOCKSIZE)):
            stream.read(BLOCKSIZE)
        print("Calibrating background noise, stay quiet")
        levels = []
        for _ in range(int(seconds * SAMPLERATE / BLOCKSIZE)):
            chunk, _ = stream.read(BLOCKSIZE)
            levels.append(self.rms(chunk[:,0]))
        threshold = max(np.mean(levels) * SILENCE_MARGIN, MIN_THRESHHOLD)
        print(f"Speech threshold: {threshold:.0f}")
        return threshold

    def flush(self, stream):
        n = stream.read_available
        if n:
            stream.read(n)

    def record_until_silence(self, stream, threshold, no_speech_timeout=NO_SPEECH_TIMEOUT):
        # recording chunks until you stop talking and recognises silence to switch to sleep
        chunks = []
        heard_speech = False
        silent_chunks = 0
        chunks_per_second = SAMPLERATE / BLOCKSIZE
        max_chunks = int(MAX_RECORD_SECONDS * chunks_per_second)
        silence_limit = int(SILENCE_SECONDS * chunks_per_second)
        no_speech_limit = int(no_speech_timeout * chunks_per_second)

        for i in range(max_chunks):
            chunk, _ = stream.read(BLOCKSIZE)
            chunk = chunk[:,0]
            chunks.append(chunk)

            if self.rms(chunk) > threshold:
                heard_speech = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            if heard_speech and silent_chunks >= silence_limit:
                break
            if not heard_speech and i >= no_speech_limit:
                return None

        return np.concatenate(chunks)

    def transcribe(self, audio_int16):
        audio = audio_int16.astype(np.float32)
        audio -= audio.mean()
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.7
        segments, info = self._whisper_model.transcribe(audio, language=LANGUAGE)
        return " ".join(segment.text for segment in segments).strip()

    def matches(self, text, phrases):
        cleaned = text.lower().strip(string.punctuation + " ")
        words = cleaned.split()

        return cleaned in phrases or (len(words) <= 4 and any(p in cleaned for p in phrases))

    def is_sleep_command(self, text):
        return self.matches(text, SLEEP_PHRASES)

    def is_shutdown_command(self, text):
        return self.matches(text, SHUTDOWN_PHRASES)

    def shutdown(self):
        self.say("Goodbye, my friend!")
        self.speech_q.join()
        self.stop.set()

    def say(self, text):
        text = clean_for_speech(text)
        if text:
            self.speech_q.put(text)

    def _history_chars(self):
        return sum(len(m.get("content") or "") for m in self._messages[1:])

    def trim_history(self):
        while len(self._messages) - 1 > MAX_MESSAGES or self._history_chars() > MAX_CHARS:
            nxt = next((i for i in range(2, len(self._messages))
                        if self._messages[i]["role"] == "user"), None)
            if nxt is None:
                break                                  
            self._dropped.extend(self._messages[1:nxt])
            del self._messages[1:nxt]               

    def update_summary(self):
        transcript = "\n".join(
                                f"{'User' if m['role'] == 'user' else 'BMO'}: {m['content']}"
                                for m in self._dropped
                                if m["role"] in ("user", "assistant") and m.get("content")
        )
        if not transcript:
            self._dropped = []
            return
        prompt = (
            "Below is a summary of an earlier conversation, followed by newer dialogue. "
            "Write an updated summary in at most 5 sentences. Keep only lasting facts: "
            "the user's name, preferences, plans and promises. Skip small talk.\n\n"
            f"Current summary: {self._summary or '(none)'}\n\nNew dialogue:\n{transcript}\n\nUpdated summary:"
        )
        try:
            r = chat(model=SUMMARY_MODEL, messages=[{"role": "user", "content": prompt}])
            self._summary = r.message.content.strip()
        except Exception:
            traceback.print_exc()
            return                                  
        self._dropped = []
        self._messages[0]["content"] = (
            self._personality + "\n\nWhat you remember from earlier:\n" + self._summary
        )

    def responding(self, prompt):
        self._messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        self.trim_history()

        full_reply = ""
        for _ in range(MAX_TOOL_ROUNDS):
            stream = chat(
                model=self._ollama_model,
                messages=self._messages,
                tools=list(self._tools.values()) or None,
                stream=True,
            )

            text, buf, tool_calls = "", "", []
            for chunk in stream:
                piece = chunk.message.content or ""
                text += piece
                buf += piece
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

                while True:
                    m = SENTENCE_END.search(buf, MIN_SENTENCE)
                    if not m:
                        break
                    self.say(buf[:m.end()])
                    buf = buf[m.end():]
            if buf.strip():
                self.say(buf)

            msg = {"role": "assistant", "content": text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                print(msg["tool_calls"])
            self._messages.append(msg)
            full_reply += text

            if not tool_calls:
                break

            for call in tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                func = self._tools.get(name)
                try:
                    result = func(**args) if func else f"Unknown tool: {name}"
                except Exception as e:
                    traceback.print_exc()
                    result = f"Tool failed: {e}"
                print(f"[tool] {name}({args}) -> {result}")
                self._messages.append({"role": "tool", "tool_name": name, "content": str(result)})

        return full_reply

    # Main funktions

    def voice_loop(self):
        if self.stop.is_set():
            return
        try:
            with sd.InputStream(samplerate=SAMPLERATE, channels=1, blocksize=BLOCKSIZE, dtype="int16") as stream:
                threshold = self.calibrate(stream)
                buffer = np.zeros(WAKE_WINDOW, dtype=np.int16)
                frame_count = 0
                self.state = SLEEPING

                while not self.stop.is_set():
                    #SLEEP
                    self.state = "bored" if music.playing() else "sleeping"
                    chunk, _ = stream.read(BLOCKSIZE)
                    chunk = chunk[:,0]
                    buffer = np.concatenate((buffer[len(chunk):], chunk))

                    frame_count += 1
                    if frame_count % SCORE_EVERY != 0:
                        continue
                    scores = self._wake_model.predict(buffer)
                    if scores.get(WAKE_WORD, 0) <= WAKE_THRESHOLD:
                        continue

                    #AWAKE
                    print("woke up")
                    music.duck()
                    self.conversation(stream, threshold)
                    music.restore()
                    self.state = "bored" if music.playing() else "sleeping"
                    self.update_summary
                    self.flush(stream)
                    buffer[:] = 0

        except Exception:
            traceback.print_exc()

    def conversation(self, stream, threshold):
        timeout = NO_SPEECH_TIMEOUT
        misses = 0

        while not self.stop.is_set():
            self.state = LISTENING
            audio = self.record_until_silence(stream, threshold, timeout)
            if audio is None:
                return

            self.state = THINKING
            print("transcribing")
            text = self.transcribe(audio)
            if not text:
                misses += 1
                if misses >= 2:
                    return
                continue
            misses = 0

            if self.matches(text, STOP_MUSIC_PHRASES):
                music.stop()
                self.speech_q.join()
                return
            if self.is_shutdown_command(text):
                self.shutdown()
                return
            if self.is_sleep_command(text):
                return
            
            print("responding")
            reply = self.responding(text)
            print("BMO:", reply)

            self.speech_q.join()
            if music.started:
                music.started =  False
                return
            self.flush(stream)
            timeout = FOLLOWUP_TIMEOUT

    def synth_loop(self):
        try:
            while not self.stop.is_set():
                try:
                    text = self.speech_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    parts = [c.audio_int16_array for c in self._voice_model.synthesize(text, syn_config=self._syn_config)]
                    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
                except Exception:
                    traceback.print_exc()
                    audio = np.zeros(0, dtype=np.int16)
                self.audio_q.put(audio)
        except Exception:
            traceback.print_exc()

    def speech_loop(self):
        try:
            with sd.OutputStream(samplerate=self._rate, channels=1, dtype="int16") as out:
                while not self.stop.is_set():
                    try:
                        audio = self.audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        if len(audio):
                            self.state = SPEAKING
                            out.write(audio.reshape(-1,1))
                    finally:
                        self.speech_q.task_done()
        except Exception:
            traceback.print_exc()

# FrontEnd

class FrontEnd:
    def __init__(self, eyes:dict, mouths:dict):

        self._screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
        self._clock = pygame.time.Clock()
        self._run = True

        self._eye_sleep     = pygame.image.load(rf"{EYES_PATH}{eyes['sleeping']}").convert_alpha()
        self._eye_listen    = pygame.image.load(rf"{EYES_PATH}{eyes['listening']}").convert_alpha()
        self._eye_think     = pygame.image.load(rf"{EYES_PATH}{eyes['thinking']}").convert_alpha()
        self._eye_speak     = pygame.image.load(rf"{EYES_PATH}{eyes['speaking']}").convert_alpha()
        self._eye_bored     = pygame.image.load(rf"{EYES_PATH}{eyes['bored']}").convert_alpha()
        self._mouth_sleep   = []
        self._mouth_listen  = []
        self._mouth_think   = []
        self._mouth_speak   = []
        self._mouth_bored   = []

        for i in range(len(mouths["sleeping"])):
            self._mouth_sleep.append(pygame.image.load(rf"{MOUTH_PATH}{mouths['sleeping'][i]}").convert_alpha())
        for i in range(len(mouths["listening"])):
            self._mouth_listen.append(pygame.image.load(rf"{MOUTH_PATH}{mouths['listening'][i]}").convert_alpha())
        for i in range(len(mouths["thinking"])):
            self._mouth_think.append(pygame.image.load(rf"{MOUTH_PATH}{mouths['thinking'][i]}").convert_alpha())
        for i in range(len(mouths["speaking"])):
            self._mouth_speak.append(pygame.image.load(rf"{MOUTH_PATH}{mouths['speaking'][i]}").convert_alpha())
        for i in range(len(mouths["bored"])):
            self._mouth_bored.append(pygame.image.load(rf"{MOUTH_PATH}{mouths['bored'][i]}").convert_alpha())
        
        self._eyes = {
            "sleeping"  : self._eye_sleep,
            "listening" : self._eye_listen,
            "thinking"  : self._eye_think,
            "speaking"  : self._eye_speak,
            "bored"     : self._eye_bored
        }

        self._mouths = {
            "sleeping"  : self._mouth_sleep,
            "listening" : self._mouth_listen,
            "thinking"  : self._mouth_think,
            "speaking"  : self._mouth_speak,
            "bored"     : self._mouth_bored
        }

        self._current_mouth = None

        self._idle_change = 15000
        self._idle_next = 0
        self._speaking_change = 150
        self._speaking_next = 0
        self._last_state = None

    def run(self, get_state, stop):
        try:
            while self._run and not stop.is_set():
                self.events()
                self._clock.tick(FPS)
                self.draw(get_state())
        except KeyboardInterrupt:
            pass

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._run = False

    def eye(self, state):
        return self._eyes[state]

    def mouth(self, state):

        if state != self._last_state:
            self._last_state = state
            self._idle_next = 0
            self._speaking_next = 0

        current_time = pygame.time.get_ticks()

        if state != "speaking":

            if current_time >= self._idle_next:
                self._current_mouth = random.choice(self._mouths[state])
                self._idle_next = current_time + self._idle_change

        else:

            if current_time >= self._speaking_next:
                self._current_mouth = random.choice(self._mouths[state])
                self._speaking_next = current_time + self._speaking_change

        return self._current_mouth

    def draw(self, state):
        self._screen.fill("#d9ffea")
        self._screen.blit(self.eye(state), (0,0))
        self._screen.blit(self.mouth(state), (0,0))
        pygame.display.update()

# BMO

class BMO:
    def __init__(self, models:dict, personality:str, eyes:dict, mouths:dict, tools: dict=None):
        self.BackEnd = BackEnd(models, personality, TOOLS)
        self.FrontEnd = FrontEnd(eyes, mouths)

    def face_state(self):
        return self.BackEnd.state

    def run(self):
        if self.BackEnd.stop.is_set():
            return
        threading.Thread(target=self.BackEnd.voice_loop, daemon=True).start()
        threading.Thread(target=self.BackEnd.speech_loop, daemon=True).start()
        threading.Thread(target=self.BackEnd.synth_loop, daemon=True).start()

        self.FrontEnd.run(self.face_state, self.BackEnd.stop)

        self.BackEnd.stop.set()
        pygame.quit()

# Running the program

bmo = BMO(models, personality, eyes, mouths)

try:
    bmo.run()
finally:
    bmo.BackEnd.stop.set()
