import pygame
import random
import numpy as np

# Initialisiere Pygame
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EEG Feedback")

# 4-Klassen der Modelvorhersage
class_names = ['Left', 'Right', 'Feet', 'Tongue']
n_classes = len(class_names)

# Position der Balken und Abstände
margin = 50
bar_width = 50
spacing = (WIDTH - 2 * margin) // n_classes

# Schriftart für Text
font = pygame.font.SysFont('Arial', 20)

# Dummy-Probs später ersetzen durch echte Vorhersagen
def dummy_probs():
    raw = np.random.rand(n_classes)
    return raw / np.sum(raw)

# "Game"-Schleife
running = True
clock = pygame.time.Clock()

while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Hol Wahrscheinlichkeiten
    probs = dummy_probs()

    for i, prob in enumerate(probs):
        # Position und Höhe
        x = margin + i * spacing
        bar_height = int(prob * (HEIGHT - 100))
        y = HEIGHT - bar_height - 50

        # Balken zeichnen
        pygame.draw.rect(screen, (100, 200, 100), (x, y, bar_width, bar_height))

        # Text unter Balken
        label = font.render(f"{class_names[i]}: {prob * 100:.2f}%", True, (255, 255, 255))
        screen.blit(label, (x - 10, HEIGHT - 40))

    pygame.display.flip()
    clock.tick(2)  # 2 Updates pro Sekunde

pygame.quit()
