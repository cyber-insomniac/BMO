from settings import *
from games import *
import pygame
import random
import threading
import queue
from pathlib  import Path
import subprocess
import json
import socket
import os
import time
pygame.init()

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

