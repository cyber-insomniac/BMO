import pygame
import math
import random
pygame.init()
WIDTH = 1024
HEIGHT = 600
FPS = 60
FONT = pygame.font.Font(None, 72)

class Pong:
    def __init__(self):
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
        self._clock = pygame.time.Clock()
        self._run = True
        
        self._player1_rect = pygame.Rect((10, HEIGHT // 2), (50, 200))
        self._player2_rect = pygame.Rect((964, HEIGHT // 2), (50, 200))
        self._ball_speed = 10
        self._ball_radius = 20
        self._player_speed = 5
        self._comp1_speed = 2
        self._comp2_speed = 3
        self._size = (50,50)
        self._direction_x = 1
        self._direction_y = 1
        self._ball_x = 0
        self._ball_y = 0
        self._score_1 = 0
        self._score_2 = 0
        self._aim_error = random.randint(-40, 40)

    def run(self, mode: str = None):
        try:
            while self._run:
                self.events()
                self.update(mode)
                self._clock.tick(FPS)
                self.draw()
        except KeyboardInterrupt:
            pass

    def update(self, mode: str = None):
        if mode == "player":
            self.player()
        else:
            self.dumb_comp()
        self.smart_comp()
        self.ball()

    def dumb_comp(self):
        if self._player1_rect.y != self._ball_y:
            if self._player1_rect.y >= 0 or self._player1_rect.y <= HEIGHT:
                if self._direction_y <= 0:
                    self._player1_rect.y -= self._comp2_speed
                else:
                    self._player1_rect.y += self._comp2_speed
        self._player1_rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

    def fold(self, y, top, bottom):
        span = bottom - top
        y = (y - top) % (2 * span)
        if y > span:
            y = 2 * span - y
        return y + top

    def smart_comp(self):
        if self._direction_x > 0:
            dx = self._player2_rect.left - self._ball_radius - self._ball_x
            raw_y = self._ball_y + (self._direction_y / self._direction_x) * dx
            target = self.fold(raw_y, self._ball_radius, HEIGHT - self._ball_radius)
        else:
            target = HEIGHT / 2

        target += self._aim_error
        diff = target - self._player2_rect.centery
        self._player2_rect.centery += max(-self._comp1_speed, min(self._comp1_speed, diff))
        self._player2_rect.clamp_ip(pygame.Rect(0,0, WIDTH, HEIGHT))

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
        if self._ball_x <= 0:
            self._score_2 += 1

        if self._ball_x >= 1024:
            self._score_1 += 1

        #print(f"player 1: {self._score_1} \nplayer 2: {self._score_2}")

    def collide(self):
        # Y
        if self._ball_y - self._ball_radius <= 0:
            self._ball_y = self._ball_radius
            self._direction_y = abs(self._direction_y)
        elif self._ball_y + self._ball_radius >= 600:
            self._ball_y = HEIGHT - self._ball_radius
            self._direction_y = -abs(self._direction_y)

        # Player
        if self._player1_rect.collidepoint((self._ball_x - self._ball_radius), self._ball_y):
            self._direction_x = abs(self._direction_x)
        elif self._player2_rect.collidepoint((self._ball_x + self._ball_radius), self._ball_y):
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

        self._ball_x += self._direction_x * self._ball_speed
        self._ball_y += self._direction_y * self._ball_speed



    def draw(self):
        self._screen.fill("#d9ffea")
        pygame.draw.rect(self._screen, "#63bda4", self._player1_rect)
        pygame.draw.rect(self._screen, "#63bda4", self._player2_rect)
        pygame.draw.circle(self._screen, "#63bda4", (self._ball_x, self._ball_y), self._ball_radius)
        self._screen.blit(FONT.render(str(self._score_1), True, "#63bda4"), ((WIDTH // 2) - 50 , 50))
        self._screen.blit(FONT.render(str(self._score_2), True, "#63bda4"), ((WIDTH // 2) + 50 , 50))
        pygame.display.update()

g = Pong()
g.run("player")
