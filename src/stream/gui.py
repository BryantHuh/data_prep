import os
import time
import pygame
import torch
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter
from braindecode.preprocessing import exponential_moving_standardize

# --- Konfiguration ---
WINDOW_SIZE = 500
WINDOW_STRIDE = 25
N_CHANNELS = 16
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/moabb_downsampled_good_subjects_model_full.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
colors = [(0, 102, 204), (0, 153, 76), (255, 153, 0), (204, 0, 102)]
labels_text = ['left', 'right', 'feet', 'tongue']

# --- Voting-Mechanismus ---
voting_window = []
VOTING_HISTORY_LENGTH = 10

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BCI Feedback")
font = pygame.font.SysFont("Arial", 30)

# --- LSL-Stream Finden ---
def wait_for_eeg_stream(timeout=60, retry_interval=1):
    print("Suche nach EEG-Stream...")
    start_time = time.time()
    while True:
        streams = resolve_byprop('type', 'EEG', timeout=retry_interval)
        if streams:
            print("EEG-Stream gefunden.")
            return StreamInlet(streams[0])
        elif time.time() - start_time > timeout:
            raise TimeoutError("Kein EEG-Stream gefunden.")

# --- Modell laden ---
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

# --- Ringbuffer ---
ring_buffer = np.zeros((0, N_CHANNELS))

def update_buffer(new_sample):
    global ring_buffer
    ring_buffer = np.vstack([ring_buffer, new_sample])
    if ring_buffer.shape[0] > WINDOW_SIZE:
        ring_buffer = ring_buffer[-WINDOW_SIZE:]

# --- Preprocessing ---
def butter_bandpass(lowcut=4, highcut=38, fs=125, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def apply_filter(data, b, a):
    return lfilter(b, a, data, axis=0)

# --- Sliding-Window Prediction ---
def predict_sliding_windows(buffer, model, device):
    if buffer.shape[0] < WINDOW_SIZE:
        return None

    b, a = butter_bandpass()
    filtered = apply_filter(buffer, b, a)
    standardized = exponential_moving_standardize(filtered, factor_new=0.001, init_block_size=100)

    windows = [standardized[i:i + WINDOW_SIZE].T for i in range(0, standardized.shape[0] - WINDOW_SIZE + 1, WINDOW_STRIDE)]
    if not windows:
        return None

    inputs = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(inputs)
        probs = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        return probs.mean(axis=(0, 2))

# --- LSL Start ---
try:
    inlet = wait_for_eeg_stream(timeout=30)
except TimeoutError as e:
    print(e)
    pygame.quit()
    exit(1)

# --- Hauptloop ---
running = True
clock = pygame.time.Clock()

while running:
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    try:
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is not None:
            update_buffer(np.array(sample)[np.newaxis, :N_CHANNELS])
    except RuntimeError:
        print("LSL-Stream-Verbindung verloren.")
        running = False
        break

    new_probs = predict_sliding_windows(ring_buffer, model, device)
    if new_probs is not None:
        voting_window.append(new_probs)
        if len(voting_window) > VOTING_HISTORY_LENGTH:
            voting_window.pop(0)
        avg_probs = np.mean(voting_window, axis=0)
    else:
        avg_probs = None

    if avg_probs is not None and len(avg_probs) == 4:
        for i in range(4):
            x = i * (SCREEN_WIDTH // 4) + 50
            p = float(avg_probs[i])
            bar_height = int(p * SCREEN_HEIGHT)
            pygame.draw.rect(screen, colors[i], (x, SCREEN_HEIGHT - bar_height, 50, bar_height))
            label = font.render(f"{labels_text[i]}: {p:.2f}", True, (0, 0, 0))
            screen.blit(label, (x, SCREEN_HEIGHT - bar_height - 30))
    else:
        for i in range(4):
            x = i * (SCREEN_WIDTH // 4) + 50
            pygame.draw.rect(screen, (200, 200, 200), (x, SCREEN_HEIGHT - 50, 50, 50))
            screen.blit(font.render("...", True, (100, 100, 100)), (x, SCREEN_HEIGHT - 100))

    pygame.display.flip()
    clock.tick(125)

pygame.quit()
