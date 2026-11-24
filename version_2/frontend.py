from settings import *
from tools import *
import pygame
import random
import threading
import queue
import traceback
pygame.init()

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

