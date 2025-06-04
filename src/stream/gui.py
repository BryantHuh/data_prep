import os
import pygame
import torch
import numpy as np
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet

# -----------------------------------
# 1. Konfiguration
# -----------------------------------
WINDOW_SIZE = 500  # Modell erwartet 500 Zeitschritte (4 Sekunden bei 125 Hz)
N_CHANNELS = 16 # Anzahl der Kanäle
# (C3, C4, Cz, FC1, FC2, FCz, CP1, CP2, CPz, P1, P2, Pz, C1, C2, CP3, CP4)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/moabb_downsampled_good_subjects_model_full.pth') # Pfad zum Modell WICHTIG anpassen wenn anderer Pfad verwendet wird!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------
# 2. Initialisierung
# -----------------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BCI Feedback")
font = pygame.font.SysFont("Arial", 30)

# -----------------------------------
# 3. Modell laden
# -----------------------------------
model = torch.load(MODEL_PATH, map_location=torch.device(device), weights_only=False) #weights_only=False ist wichtig, damit die Architektur geladen wird und nicht nur die Gewichte
model.eval()

# -----------------------------------
# 4. Ringbuffer initialisieren
# -----------------------------------
ring_buffer = np.zeros((0, N_CHANNELS))  # Start mit leerem Puffer

# Funktion zum Aktualisieren des Ringbuffers
# Hier wird der Ringbuffer erweitert und auf die maximale Größe begrenzt (500)
def update_buffer(new_data):
    global ring_buffer
    ring_buffer = np.concatenate([ring_buffer, new_data], axis=0)
    if len(ring_buffer) > WINDOW_SIZE:
        ring_buffer = ring_buffer[-WINDOW_SIZE:]  # Nur die letzten WINDOW_SIZE behalten

# -----------------------------------
# 5. Vorhersagefunktion
# -----------------------------------
def predict_from_buffer(buffer):
    if buffer.shape[0] < WINDOW_SIZE:
        return None  # Noch nicht genug Daten (< 500 Zeitschritte)
    standardized = exponential_moving_standardize(buffer, factor_new=0.001, init_block_size=100)
    input_tensor = torch.tensor(standardized.T[np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
        #print("Raw model output shape:", pred.shape)
        probs = torch.nn.functional.softmax(pred, dim=1).mean(dim=2).cpu().numpy()[0] # Durchschnitt über die Zeitachse
        return probs

# -----------------------------------
# 6. Hauptloop
# -----------------------------------
running = True
clock = pygame.time.Clock()

while running:
    screen.fill((255, 255, 255))


    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Fenster schließen mit klicken auf das X
            running = False

    # EEG-Daten simulieren (z. B. 10 Samples pro Frame)
    #TODO: Hier sollten echte EEG-Daten verwendet werden
    dummy_data = np.random.randn(10, N_CHANNELS)
    update_buffer(dummy_data)

    probs = predict_from_buffer(ring_buffer)
    #probs = np.random.dirichlet(np.ones(4))  # Simulierte Vorhersage für 4 Klassen
    #print(f"Vorhersage: {probs}")

    #if probs is not None:
    #    print(len(probs), probs)
    # Balkendiagramm zeichnen
    if probs is not None and len(probs) == 4: # Sicherstellen, dass 4 Klassen vorhanden sind
        for i in range(4):
            x = i * (SCREEN_WIDTH // 4) + 50
            p = float(probs[i])
            bar_height = int(p * SCREEN_HEIGHT)
            pygame.draw.rect(screen, (0, 0, 255), (x, SCREEN_HEIGHT - bar_height, 50, bar_height))
            label = font.render(f"{p:.2f}", True, (0, 0, 0))
            screen.blit(label, (x, SCREEN_HEIGHT - bar_height - 30))
    else:
        for i in range(4): # Wenn nicht genug Daten, Platzhalter zeichnen
            x = i * (SCREEN_WIDTH // 4) + 50
            pygame.draw.rect(screen, (200, 200, 200), (x, SCREEN_HEIGHT - 50, 50, 50))
            label = font.render("...", True, (100, 100, 100))
            screen.blit(label, (x, SCREEN_HEIGHT - 100))

    pygame.display.flip() # Bildschirm aktualisieren
    clock.tick(10) # 10 FPS

pygame.quit()
