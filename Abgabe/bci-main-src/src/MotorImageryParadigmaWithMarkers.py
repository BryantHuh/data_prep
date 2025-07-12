import pygame
import sys
from pylsl import StreamInfo, StreamOutlet
import time
import random

# Initialisiere LSL
info = StreamInfo('MarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
outlet = StreamOutlet(info)

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Pygame starten
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("EEG Stimulus")
font = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()

# Texteingabe & Button Setup
input_box = pygame.Rect(300, 200, 200, 50)
button_box = pygame.Rect(350, 300, 100, 50)
user_text = ''
start_experiment = False

# Bildschirm-Mitte
cx, cy = 400, 300

# Basis-Pfeil
base = [(-100, 0), (0, -50), (0, -20), (100, -20), (100, 20), (0, 20), (0, 50)]

def rotate(pt, angle):
    from math import cos, sin, radians
    x, y = pt
    a = radians(angle)
    return (x * cos(a) - y * sin(a), x * sin(a) + y * cos(a))

points_left = [(cx + x, cy + y) for x, y in base]
points_up = [(cx + x, cy + y) for x, y in (rotate(p, 90) for p in base)]
points_right = [(cx + x, cy + y) for x, y in (rotate(p, 180) for p in base)]
points_down = [(cx + x, cy + y) for x, y in (rotate(p, -90) for p in base)]

def draw_text_center(text, y):
    rendered = font.render(text, True, WHITE)
    rect = rendered.get_rect(center=(400, y))
    screen.blit(rendered, rect)

def draw_fixation_cross():
    pygame.draw.line(screen, WHITE, (200, 300), (600, 300), 2)
    pygame.draw.line(screen, WHITE, (400, 100), (400, 500), 2)

def draw_arrow(direction):
    if direction == 'left':
        pygame.draw.polygon(screen, WHITE, points_left)
    elif direction == 'right':
        pygame.draw.polygon(screen, WHITE, points_right)
    elif direction == 'up':
        pygame.draw.polygon(screen, WHITE, points_up)
    elif direction == 'down':
        pygame.draw.polygon(screen, WHITE, points_down)

# Zeitsteuerung
experiment_state = 'idle'
marker_sent = False
directions = []
current_index = 0
state_start_time = 0

# Hauptloop
while True:
    screen.fill(BLACK)
    now = pygame.time.get_ticks()  # Zeit in Millisekunden

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if experiment_state == 'idle':
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    user_text += event.unicode
            if event.type == pygame.MOUSEBUTTONDOWN and button_box.collidepoint(event.pos):
                try:
                    runs_left = int(user_text)
                    if runs_left % 4 != 0:
                        user_text = ''
                    else:
                        # Zufällige aber gleichverteilte Richtungen
                        directions = ['left', 'right', 'up', 'down'] * (runs_left // 4)
                        random.shuffle(directions)
                        current_index = 0
                        experiment_state = 'fixation'
                        state_start_time = now
                except:
                    user_text = ''

    # Experiment-Ablauf
    if experiment_state == 'fixation':
        draw_fixation_cross()
        if now - state_start_time >= 2000:
            experiment_state = 'arrow'
            state_start_time = now

    elif experiment_state == 'arrow':
        direction = directions[current_index]
        draw_arrow(direction)

        # Marker nur einmal senden
        if not marker_sent:
            outlet.push_sample([direction])
            print(f"Marker gesendet: {direction}")
            marker_sent = True

        if now - state_start_time >= 4000:
            current_index += 1
            if current_index < len(directions):
                experiment_state = 'fixation'
                state_start_time = now
                marker_sent = False  # Zurücksetzen für nächsten Pfeil
            else:
                experiment_state = 'idle'
                user_text = ''


    # UI bei Stillstand
    if experiment_state == 'idle':
        pygame.draw.rect(screen, WHITE, input_box, 2)
        text_surface = font.render(user_text, True, WHITE)
        screen.blit(text_surface, (input_box.x + 10, input_box.y + 10))
        draw_text_center("Anzahl der Durchläufe (durch 4 teilbar):", 150)

        pygame.draw.rect(screen, WHITE, button_box, 2)
        draw_text_center("Start", 325)

    pygame.display.flip()
    clock.tick(30)
