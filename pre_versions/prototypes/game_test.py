import pygame
import math
import random
pygame.init()
WIDTH = 1024
HEIGHT = 600
FPS = 60


class Pong:
    def __init__(self):
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
        self._clock = pygame.time.Clock()
        self._run = True
        
        self._player1_rect = pygame.Rect((10, HEIGHT // 2), (50, 200))
        self._player2_rect = pygame.Rect((964, HEIGHT // 2), (50, 200))
        self._ball_speed = 10
        self._player_speed = 5
        self._comp1_speed = 4
        self._comp2_speed = 3
        self._size = (50,50)
        self._direction_x = 1
        self._direction_y = 1
        self._ball_x = 0
        self._ball_y = 0
        self._score_1 = 0
        self._score_2 = 0

    def run(self, mode):
        try:
            while self._run:
                self.events()
                self.update(mode)
                self._clock.tick(FPS)
                self.draw()
        except KeyboardInterrupt:
            pass

    def update(self, mode):
        if mode == "player":
            self.player()
        else:
            pass
        self.dumb_comp()
        self.ball()

    def dumb_comp(self):
        if self._player2_rect.y != self._ball_y:
            if self._direction_y <= 0:
                self._player2_rect.y -= self._comp2_speed
            else:
                self._player2_rect.y += self._comp2_speed


    def player(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self._player1_rect.y -= self._player_speed
        if keys[pygame.K_DOWN]:
            self._player1_rect.y += self._player_speed

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._run = False

    def direction(self):
        angle = random.randint(-45, 45)
        self._direction_x = math.cos(math.radians(angle))
        self._direction_y = math.sin(math.radians(angle))

    def reset_ball(self):
        self._ball_x = WIDTH // 2
        self._ball_y = HEIGHT // 2
        self.direction()

    def score(self):
        if self._ball_x == 0:
            self._score_1 += 1

        if self._ball_x == 1024:
            self._score_2 += 1

        print(f"player 1: {self._score_1} \nplayer 2: {self._score_2}")

    def collide(self):
        # Y
        if self._ball_y - 20 <= 0:
            self._ball_y = 20
            self._direction_y = abs(self._direction_y)
        elif self._ball_y + 20 >= 600:
            self._ball_y = 580
            self._direction_y = -abs(self._direction_y)

        # Player
        if self._player1_rect.collidepoint((self._ball_x - 20), self._ball_y):
            #self._ball_x = 30
            self._direction_x = abs(self._direction_x)
        elif self._player2_rect.collidepoint((self._ball_x + 20), self._ball_y):
            #self._ball_x = 994
            self._direction_x = -abs(self._direction_x)

    def ball(self):
        current_time = pygame.time.get_ticks()
        if current_time <= 100:
            self.reset_ball()
            return

        if self._ball_x <= 0 or self._ball_x >= 1024:
            self.score()
            self.reset_ball()
            return

        self.collide()

        self._ball_x += self._direction_x * self._ball_speed
        self._ball_y += self._direction_y * self._ball_speed



    def draw(self):
        self._screen.fill("#d9ffea")
        pygame.draw.rect(self._screen, "#63bda4", self._player1_rect)
        pygame.draw.rect(self._screen, "#63bda4", self._player2_rect)
        pygame.draw.circle(self._screen, "#63bda4", (self._ball_x, self._ball_y), 20)
        pygame.display.update()

g = Pong()
g.run("player")
