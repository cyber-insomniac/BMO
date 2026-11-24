"""
Program structure
BMO has six states currently, this is done to better differentiate between when BMO is supposed to listen or act

Sleep:
BMO doesnt actively listen/record, he takes chunks of audio and looks if his wakeword has been said (Hey BMO)

Listening:
BMO starts actively listening/recording. If he finds a prolonged period of silence he stops and moves on.

Thinking:
BMO transcribes the audio he recorded and hands it over to his ollama model. If a tool is needed he will call the tool.

Speaking:
BMO speaks his answer.

Noisy:
When BMO is asked to perform a task like playing music or a video

Bored:
BMO has been left alone for a period of time and thus is bored and begins to watch videos, listen to music or play games.

Conversation loop:
- waits for wakeword and counts down until BMO gets bored
  - when bored will do random actions
- listens
  - when only silence goes back to sleep
- transcribes and thinks
  - if told to shut down he terminate
  - if told goodbye he will go back to sleep
  - calls tool if needed
- speaks
- waits to see if theres more to listen to otherwise goes to sleep

"""

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
from ollama import chat
from ollama import ChatResponse
import string
from piper import PiperVoice, SynthesisConfig
import threading, queue, traceback, time, re, random, pygame
from pathlib import Path
import re
import subprocess
import os
import json
import socket
import math
pygame.init()
# Settings

BASE = Path(__file__).parent
ROOT = BASE.parent

SAMPLERATE          = 16000
BLOCKSIZE           = 1280                                  # 80 ms per chunk becaus of openwakeword
WAKE_WORD_FILE      = ROOT / "sound_models" / "hey_BMO.onnx"# Self trained wakeword model
WAKE_WORD           = ROOT / "sound_models" / "hey_BMO"
VOICE_FILE          = ROOT / "sound_models" / "en_US-amy-medium.onnx"
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
VIDEO_DUCK          = 50
VIDEO_NORMAL        = 100
BORED_TIMER         = 10                                    # Debug timer for testing
BORED_MIN           = 180
BORED_MAX           = 300

EYES_PATH           = ROOT / "assets" / "eyes"             # Asset paths for BMOs face
MOUTH_PATH          = ROOT / "assets" / "mouth"
SONG_PATH           = ROOT / "music"
VIDEO_PATH          = ROOT / "videos"
WIDTH               = 1024                                  # Screen size for pygame
HEIGHT              = 600
FPS                 = 60
FONT                = pygame.font.Font(None, 72)
SLEEPING            = "sleeping"                            # BMOs states
LISTENING           = "listening"
THINKING            = "thinking"
SPEAKING            = "speaking"
BORED               = "bored"
NOISY               = "noisy"
SLEEP_PHRASES       = ("stop", "go to sleep", "good night", "goodnight")
SHUTDOWN_PHRASES    = ("shut down", "shutdown", "power off", "turn off", "exit", "kill yourself")
STOP_MUSIC_PHRASES  = ("stop the music", "stop music", "music off", "turn off the music", "turn the music off", "pause the music", "stop the radio", "turn off the radio", "radio off")
STOP_VIDEO_PHRASES  = ("stop the video", "stop video", "video off", "turn off the video", "close the video")
STOP_GAME_PHRASES   = ("stop game", "stop the game", "quit the game", "end the game", "game over")
FILE_NAMES          = {".wav", ".mp4"}
BORED_PHRASES       = ["Hello darkness my old friend",
                       "NOOOOO dont leave",
                       "bastard",
                       "HELP",
                       "Going to goon",
                       "bruthaaaa come back",
                       ]

class Bag:
    def __init__(self, items):
        self.items, self.bag = list(items), []
    def draw(self):
        if not self.bag:
            self.bag = random.sample(self.items, len(self.items))
        return self.bag.pop()

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
    "wake"      : WakeWordModel(models=[str(WAKE_WORD_FILE)]),
    "whisper"   : WhisperModel("small", device="cpu", compute_type="int8"),
    "ollama"    : "llama3.2",
    "voice"     : PiperVoice.load(str(VOICE_FILE)),
    "summary"   : "llama3.2"
}

eyes = {
    SLEEPING    : "normal.png",
    LISTENING   : "listen.png",
    THINKING    : "think.png",
    SPEAKING    : "speak.png",
    BORED       : "bored.png",
    NOISY       : "bored.png"

}

mouths = {
    SLEEPING    : ["normal1.png", "normal2.png", "normal3.png"],
    LISTENING   : ["listen1.png"],
    THINKING    : ["think1.png"],
    SPEAKING    : ["speak1.png", "speak2.png", "speak3.png", "speak4.png", "speak5.png", "speak6.png"],
    BORED       : ["bored1.png", "bored2.png"],
    NOISY       : ["bored1.png", "bored2.png"]
}

# Tools

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

class Music:
    def __init__(self, folder=SONG_PATH):
        pygame.mixer.init()
        self.files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in FILE_NAMES)
        self.current = None
        self.started = False
        self.radio = False
        self.volume = NORMAL_VOL
        self._queue = []
        self._lock = threading.Lock()
        threading.Thread(target=self.radio_loop, daemon=True).start()

    def load(self, path):
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.set_volume(self.volume)
        pygame.mixer.music.play()
        self.current = path

    def start_next(self):
        if not self._queue:
            self._queue = random.sample(self.files, len(self.files))
            if len(self._queue) > 1 and self._queue[0] == self.current:
                self._queue.append(self._queue.pop(0))
        try:
            self.load(self._queue.pop(0))
        except Exception:
            traceback.print_exc()

    def radio_loop(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                if self.radio and not pygame.mixer.music.get_busy():
                    self.start_next()

    def start_radio(self):
        if not self.files:
            return "There are no music files in the music folder"
        videos.stop()
        with self._lock:
            self.radio = True
            self._queue = []
            self.start_next()
            self.started = True
        return f"Radio mode is on! First up: {self._describe(self.current)}"

    def _describe(self, path):
        name = path.stem.replace("_", " ").strip()
        if " - " in name:
            artist, title = (p.strip() for p in name.split(" - ", 1))
            return f"{title} by {artist}"
        return name 

    def playing(self):
        return pygame.mixer.music.get_busy() or self.radio

    def play(self, query):
        if not self.files:
            return "There are no music files in the music folder."
        q = _norm(query)
        matches = [f for f in self.files if q and q != "random" and q in _norm(f.stem)]
        path = random.choice(matches or self.files)
        with self._lock:
            self.radio = False
            self.load(path)
            self.started = True
        note = "" if matches or q in ("", "random") else "I could not find that song, so I picked another. "
        return f"{note}Now playing: {self._describe(path)}"

    def now_playing(self):
        return f"Now playing: {self._describe(self.current)}" if self.playing() and self.current else "No music is playing."

    def stop(self):
        with self._lock:
            self.radio = False
            was_playing = self.playing()
            pygame.mixer.music.stop()
        return "The music is stopped." if was_playing else "No music was playing."

    def duck(self):
        self.volume = DUCK_VOL
        if self.playing():
            pygame.mixer.music.set_volume(self.volume)

    def restore(self):
        self.volume = NORMAL_VOL
        if self.playing():
            pygame.mixer.music.set_volume(self.volume)

music = Music()

def play_radio() -> str:
    """Be a radio: play random songs one after another until told to stop."""
    return music.start_radio()

def play_music(song: str) -> str:
    """Start playing a song from BMO's music folder.

    Args:
        song: Part of the song name to play, or the word random for any song.
    """
    return music.play(song)

def stop_music() -> str:
    """Stop the music that is playing."""
    return music.stop()

class Videos:
    def __init__(self, folder=VIDEO_PATH):
        self.files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in FILE_NAMES)
        self.current = None
        self.started = False
        self.process = None
        self.ipc_path = "/tmp/bmo_mpv.sock"

    def _describe(self, path):
        name = path.stem.replace("_", " ").strip()
        if " - " in name:
            artist, title = (p.strip() for p in name.split(" - ", 1))
            return f"{title} by {artist}"
        return name 
    
    def playing(self):
        return self.process is not None and self.process.poll() is None

    def play(self, query):
        if not self.files:
            return "There are no video files in the video folder."
        music.stop()
        self.stop()
        q = _norm(query)
        matches = [f for f in self.files if q and q != "random" and q in _norm(f.stem)]
        path = random.choice(matches or self.files)
        try:
            os.unlink(self.ipc_path)
        except FileNotFoundError:
            pass
        self.process = subprocess.Popen([
            "mpv",
            "--fullscreen",
            "--ontop",
            "--no-border",
            "--no-terminal",
            #"--force-window=yes",
            f"--volume={VIDEO_DUCK}",
            f"--input-ipc-server={self.ipc_path}",
            str(path)
        ])
        self.current, self.started = path, True
        note = "" if matches or q in ("", "random") else "I could not find that video, so I picked another. "
        return f"{note}Now playing: {self._describe(path)}"

    def command(self, *args):
        if not self.playing():
            return
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.settimeout(1)
                s.connect(self.ipc_path)
                s.sendall((json.dumps({"command": list(args)}) + "\n").encode())
        except OSError:
            pass

    def now_playing(self):
        if self.playing() and self.current:
            return f"Now playing: {self._describe(self.current)}"
        return "No video is playing."

    def stop(self):
        was_playing = self.playing()
        if was_playing:
            self.command("quit")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.process = None
        return "The video is stopped." if was_playing else "No video was playing."

    def duck(self):
        self.command("set_property", "volume", VIDEO_DUCK)

    def restore(self):
        self.command("set_property", "volume", VIDEO_NORMAL)

videos = Videos()

def play_video(video:str) -> str:
    """Start playing a video from BMO's video folder.

    Args:
        video: Part of the video name to play, or the word random for any song.
    """
    return videos.play(video)

def whats_playing() -> str:
    """Tell which song or video is currently playing."""
    if music.playing():
        return music.now_playing()
    if videos.playing():
        return videos.now_playing()
    return "Nothing is playing."

def stop_video() -> str:
    """Stop the video that is playing."""
    return videos.stop()

def media_playing():
    return music.playing() or videos.playing()

class Game:
    def __init__(self, who):
        self.who = who
        self.done = False
    def handle_event(self, event): pass
    def update(self, dt): pass
    def draw(self, screen): pass

class Pong(Game):
    BALL_RADIUS, RECT_W, RECT_H = 20, 50, 200
    PLAYER_SPD, COMP1_SPD, COMP2_SPD, BALL_SPD = 5, 2, 3, 10

    def __init__(self, who):
        super().__init__(who)
        self._player1_rect = pygame.Rect((10, HEIGHT // 2), (Pong.RECT_W, Pong.RECT_H))
        self._player2_rect = pygame.Rect((964, HEIGHT // 2), (Pong.RECT_W, Pong.RECT_H))
        self._score_1 = 0
        self._score_2 = 0
        self._ball_x = 0
        self._ball_y = 0
        self._direction_x = 1
        self._direction_y = 1
        self._aim_error = random.randint(-40, 40)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.done = True

    def update(self):
        if self.who == "user":
            self.player()
        else:
            self.dumb_comp()
        self.smart_comp()
        self.ball()

    def dumb_comp(self):
        if self._player1_rect.y != self._ball_y:
            if self._player1_rect.y >= 0 or self._player1_rect.y <= HEIGHT:
                if self._direction_y <= 0:
                    self._player1_rect.y -= Pong.COMP2_SPD
                else:
                    self._player1_rect.y += Pong.COMP2_SPD
        self._player1_rect.clamp_ip(pygame.Rect(0,0,WIDTH,HEIGHT))

    def fold(self, y, top, bottom):
        span = bottom - top
        y = (y - top) % (2 * span)
        if y > span:
            y = 2 * span - y
        return y + top

    def smart_comp(self):
        if self._direction_x > 0:
            dx = self._player2_rect.left - Pong.BALL_RADIUS - self._ball_x
            raw_y = self._ball_y + (self._direction_y / self._direction_x) * dx
            target = self.fold(raw_y, Pong.BALL_RADIUS, HEIGHT - Pong.BALL_RADIUS)
        else:
            target = HEIGHT / 2

        target += self._aim_error
        diff = target - self._player2_rect.centery
        self._player2_rect.centery += max(-Pong.COMP1_SPD, min(Pong.COMP1_SPD, diff))
        self._player2_rect.clamp_ip(pygame.Rect(0,0, WIDTH, HEIGHT))

    def player(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self._player1_rect.y -= Pong.PLAYER_SPD
        if keys[pygame.K_DOWN]:
            self._player1_rect.y += Pong.PLAYER_SPD

    def direction(self):
        angle = random.randint(-45, 45)
        self._direction_x = math.cos(math.radians(angle))
        self._direction_y = math.sin(math.radians(angle))

    def reset_ball(self):
        self._ball_x = WIDTH // 2
        self._ball_y = HEIGHT // 2
        self.direction()

    def score(self):
        if self._ball_x <= 0:
            self._score_2 += 1

        if self._ball_x >= 1024:
            self._score_1 += 1

    def collide(self):
        # Y
        if self._ball_y - Pong.BALL_RADIUS <= 0:
            self._ball_y = Pong.BALL_RADIUS
            self._direction_y = abs(self._direction_y)
        elif self._ball_y + Pong.BALL_RADIUS >= 600:
            self._ball_y = HEIGHT - Pong.BALL_RADIUS
            self._direction_y = -abs(self._direction_y)

        # Player
        if self._player1_rect.collidepoint((self._ball_x - Pong.BALL_RADIUS), self._ball_y):
            self._direction_x = abs(self._direction_x)
        elif self._player2_rect.collidepoint((self._ball_x + Pong.BALL_RADIUS), self._ball_y):
            self._direction_x = -abs(self._direction_x)

    def ball(self):
        current_time = pygame.time.get_ticks()
        if current_time <= 100:
            self.reset_ball()
            return

        if self._ball_x <= 0 or self._ball_x >= WIDTH:
            self.score()
            self.reset_ball()
            return

        self.collide()

        self._ball_x += self._direction_x * Pong.BALL_SPD
        self._ball_y += self._direction_y * Pong.BALL_SPD

    def draw(self, screen):
        screen.fill("#d9ffea")
        pygame.draw.rect(screen, "#63bda4", self._player1_rect)
        pygame.draw.rect(screen, "#63bda4", self._player2_rect)
        pygame.draw.circle(screen, "#63bda4", (self._ball_x, self._ball_y), Pong.BALL_RADIUS)
        screen.blit(FONT.render(str(self._score_1), True, "#63bda4"), ((WIDTH // 2) - 50 , 50))
        screen.blit(FONT.render(str(self._score_2), True, "#63bda4"), ((WIDTH // 2) + 50 , 50))
        pygame.display.update()


GAMES = {"pong": Pong}
game_requests = queue.Queue()
game_started = threading.Event()
game_running = threading.Event()

def play_game(game: str, player: str) -> str:
    """Start a small game on BMO's screen.

    Args:
        game: Which game to play. Currently only pong.
        player: Who plays: bmo to watch BMO play by himself, or user to play it yourself.
    """
    game = game.lower().strip()
    if game not in GAMES:
        return f"I only know these games: {', '.join(GAMES)}."
    who = "user" if player.lower().strip() in ("user", "me") else "bmo"
    game_requests.put((game, who))
    game_started.set()
    game_running.set()
    return f"Starting {game}. {'You play!' if who == 'user' else 'I play by myself!'}"

def stop_game() -> str:
    """Stop the game that is being played."""
    game_requests.put("quit")
    return "The game is stopped."

def take_photo():
    """Take a photo with BMO's camera and save it"""
    return

def print_sticker():
    """Print a sticker"""
    return

TOOLS = {f.__name__: f for f in (take_photo, print_sticker, play_music, play_radio, whats_playing, stop_music, play_video, stop_video, play_game, stop_game)}

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
        self._last_change       = 0
        self._last_state        = None
        self._timer             = None
        self._phrases           = Bag(BORED_PHRASES)
        self._idle_actions      = {
            "talk"      : (5, self.idle_talk),
            "music"     : (3, self.idle_music),
            "video"     : (2, self.idle_video),
            "game"      : (1, self.idle_game),
            "nothing"   : (3, lambda: None)
        }
        self.state              = SLEEPING
        self.stop = threading.Event()
        self.speech_q = queue.Queue()
        self.audio_q = queue.Queue(maxsize=3)

    # funktions
    
    def patience(self):
        self._timer = random.randint(BORED_MIN, BORED_MAX)
        print("Timer: ", self._timer)

    def busy(self):
        return (media_playing()
                or game_running.is_set()
                or self.speech_q.unfinished_tasks > 0) 

    def speak_and_wait(self, text):
        self.say(text)
        self.speech_q.join()

    def gettin_bored(self):
        if self._timer is None:
            self.patience()
        current_time = time.monotonic()
        if self.state != BORED:
            if int(current_time) - int(self._last_change) >= self._timer:
                self.change_state(BORED)
                self.patience()
        else:
            self.bored()

    def bored(self):
        if self.busy():
            return
        names = list(self._idle_actions)
        weights = [self._idle_actions[n][0] for n in names]
        name = random.choices(names, weights=weights, k=1)[0]
        self._idle_actions[name][1]()
        music.started = videos.started = False 

    def idle_talk(self):
        self.speak_and_wait(self._phrases.draw())

    def idle_music(self):
        self.speak_and_wait("I am going to play some music!")
        music.play("random")

    def idle_video(self):
        self.speak_and_wait("Let's watch something!")
        videos.play("random")

    def idle_game(self):
        self.speak_and_wait("Look at me, I'm going to play a game!")
        game_running.set()
        game_requests.put(("pong", "bmo"))

    def change_state(self, state):
        if self._last_state == BORED and state == SLEEPING:
            return
        if self._last_state != state:
            print(state)
            self.state = state
            self._last_state = state
            self._last_change = time.monotonic()

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
                self.change_state(SLEEPING)

                while not self.stop.is_set():
                    # checking if its time to get bored
                    self.gettin_bored()
                    #SLEEP
                    self.change_state(NOISY) if media_playing() and self.state != BORED else self.change_state(SLEEPING)

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
                    music.duck()
                    videos.duck()
                    self.conversation(stream, threshold)
                    music.restore()
                    videos.restore()
                    self.change_state(NOISY) if media_playing() and self.state != BORED else self.change_state(SLEEPING)
                    self.update_summary
                    self.flush(stream)
                    buffer[:] = 0

        except Exception:
            traceback.print_exc()

    def conversation(self, stream, threshold):
        timeout = NO_SPEECH_TIMEOUT
        misses = 0

        while not self.stop.is_set():
            self.change_state(LISTENING)
            audio = self.record_until_silence(stream, threshold, timeout)
            if audio is None:
                return

            self.change_state(THINKING)
            text = self.transcribe(audio)
            if not text:
                misses += 1
                if misses >= 2:
                    return
                continue
            misses = 0

            if self.matches(text, STOP_MUSIC_PHRASES + STOP_VIDEO_PHRASES + STOP_GAME_PHRASES):
                music.stop()
                videos.stop()
                game_requests.put("quit")
                self.speech_q.join()
                return
            if self.is_shutdown_command(text):
                self.shutdown()
                return
            if self.is_sleep_command(text):
                return
            
            reply = self.responding(text)
            print("BMO:", reply)

            self.speech_q.join()
            if music.started or videos.started or game_started.is_set():
                music.started =  False
                videos.started = False
                game_started.clear()
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
                            self.change_state(SPEAKING)
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
        self._game = None

        self._eye_sleep     = pygame.image.load(str(EYES_PATH / eyes[SLEEPING])).convert_alpha()
        self._eye_listen    = pygame.image.load(str(EYES_PATH / eyes[LISTENING])).convert_alpha()
        self._eye_think     = pygame.image.load(str(EYES_PATH / eyes[THINKING])).convert_alpha()
        self._eye_speak     = pygame.image.load(str(EYES_PATH / eyes[SPEAKING])).convert_alpha()
        self._eye_bored     = pygame.image.load(str(EYES_PATH / eyes[BORED])).convert_alpha()
        self._eye_noisy     = pygame.image.load(str(EYES_PATH / eyes[NOISY])).convert_alpha()
        self._mouth_sleep   = []
        self._mouth_listen  = []
        self._mouth_think   = []
        self._mouth_speak   = []
        self._mouth_bored   = []
        self._mouth_noisy   = []

        for i in range(len(mouths[SLEEPING])):
            self._mouth_sleep.append(pygame.image.load(str(MOUTH_PATH / mouths[SLEEPING][i])).convert_alpha())
        for i in range(len(mouths[LISTENING])):
            self._mouth_listen.append(pygame.image.load(str(MOUTH_PATH / mouths[LISTENING][i])).convert_alpha())
        for i in range(len(mouths[THINKING])):
            self._mouth_think.append(pygame.image.load(str(MOUTH_PATH / mouths[THINKING][i])).convert_alpha())
        for i in range(len(mouths[SPEAKING])):
            self._mouth_speak.append(pygame.image.load(str(MOUTH_PATH / mouths[SPEAKING][i])).convert_alpha())
        for i in range(len(mouths[BORED])):
            self._mouth_bored.append(pygame.image.load(str(MOUTH_PATH / mouths[BORED][i])).convert_alpha())
        for i in range(len(mouths[NOISY])):
            self._mouth_noisy.append(pygame.image.load(str(MOUTH_PATH / mouths[NOISY][i])).convert_alpha())

        
        self._eyes = {
            SLEEPING    : self._eye_sleep,
            LISTENING   : self._eye_listen,
            THINKING    : self._eye_think,
            SPEAKING    : self._eye_speak,
            BORED       : self._eye_bored,
            NOISY       : self._eye_noisy
        }

        self._mouths = {
            SLEEPING    : self._mouth_sleep,
            LISTENING   : self._mouth_listen,
            THINKING    : self._mouth_think,
            SPEAKING    : self._mouth_speak,
            BORED       : self._mouth_bored,
            NOISY       : self._mouth_noisy
        }

        self._current_mouth     = None

        self._idle_change       = 15000
        self._idle_next         = 0
        self._speaking_change   = 150
        self._speaking_next     = 0
        self._last_state        = None

    def run(self, get_state, stop):
        try:
            while self._run and not stop.is_set():
                self.events()
                self.poll_game_requests()
                if self._game:
                    events = pygame.event.get()
                    for event in events:
                        self._game.handle_event(event)
                    self._game.update()
                    self._game.draw(self._screen)
                    if self._game.done:
                        self._game = None
                        game_running.clear()
                else:
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

        if state != SPEAKING:

            if current_time >= self._idle_next:
                self._current_mouth = random.choice(self._mouths[state])
                self._idle_next = current_time + self._idle_change

        else:

            if current_time >= self._speaking_next:
                self._current_mouth = random.choice(self._mouths[state])
                self._speaking_next = current_time + self._speaking_change

        return self._current_mouth

    def poll_game_requests(self):
        try:
            while True:
                req = game_requests.get_nowait()
                if req == "quit":
                    self._game = None
                    game_running.clear()
                else:
                    name, who = req
                    self._game = GAMES[name](who)
        except queue.Empty:
            pass

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
        self.timer = BORED_TIMER

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
