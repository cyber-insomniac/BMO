import pygame
import math
import random
from settings import *
pygame.init()

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
