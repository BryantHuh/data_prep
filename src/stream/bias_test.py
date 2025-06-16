# -*- coding: utf-8 -*-
"""
Echtzeit-BCI-Feedback mit vortrainiertem ShallowFBCSPNet
- Sliding-Window-Inferenz
- Ring-Buffer (EEG-Kanäle × WINDOW_SIZE)
- EMA-Standardisierung + Bandpass
- Reset der Voting-History bei Markerwechsel
- LSL-Stream mit EEG + Marker-Kanal (nur ohne --random-only)
- Pygame-GUI: Stream, Marker, Prediction, Wahrscheinlichkeits-Balken
- Debug: Random-Input-Test + Bias-Inspektion
"""
import os
import sys
import time
import argparse

import pygame
import torch
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter
from scipy.special import softmax

from braindecode.preprocessing import exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet

# --- CLI-Argumente ---
parser = argparse.ArgumentParser(description="BCI Live Feedback mit Debug-Tools")
parser.add_argument(
    "--random-only", action="store_true",
    help="Nur den Random-Input-Test fahren und dann exit"
)
parser.add_argument(
    "--n-random", type=int, default=10,
    help="Anzahl Test-Fenster für Random-Input (default: 10)"
)
args = parser.parse_args()

# --- Konfiguration ---
N_EEG_CHANNELS     = 16
WINDOW_SIZE        = 500       # Samples (4 s @125 Hz)
VOTE_HISTORY_LENGTH= 5
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models",
    "moabb_downsampled_good_subjects_model_full.pth"
)
labels_text = ['left', 'right', 'feet', 'tongue']
colors      = [(0,102,204),(0,153,76),(255,153,0),(204,0,102)]

# --- Gerät ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Modell laden + Bias-Inspektion ---
print("Lade Modell…")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device).eval()
n_preds = model.get_output_shape()[2]
print(f"Dense-Stride (Samples): {n_preds}")

# Debug: Bias vor Reset ausgeben
print("Classifier-Bias vor Reset:")
for name, param in model.named_parameters():
    if "bias" in name:
        print(f" {name}: {param.data.cpu().numpy()}")

# Debug: alle Biases auf 0 setzen
for name, param in model.named_parameters():
    if "bias" in name:
        param.data.zero_()

print("Classifier-Bias nach Reset:")
for name, param in model.named_parameters():
    if "bias" in name:
        print(f" {name}: {param.data.cpu().numpy()}")

# --- Filter fürs EEG (Bandpass 4–38 Hz) ---
def butter_bandpass(lowcut=4, highcut=38, fs=125, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band')
b, a = butter_bandpass()

# --- Random-Input Test ---
def test_random_input(n_tests: int = 10):
    print("=== Random-Input Test ===")
    for i in range(1, n_tests + 1):
        # Weißes Rauschen (16×500)
        rnd = np.random.normal(0, 1, (N_EEG_CHANNELS, WINDOW_SIZE)).astype(np.float32)
        # identische Vorverarbeitung:
        filtered = lfilter(b, a, rnd, axis=1)
        std      = exponential_moving_standardize(
                       filtered.T, factor_new=1e-3, init_block_size=100
                   ).T
        x = torch.from_numpy(std[None]).to(device).float()
        with torch.no_grad():
            logits = model(x)                     # [1, C, T']
            probs  = softmax(logits.mean(2).cpu().numpy().squeeze())
        pred = int(probs.argmax())
        print(f"Test {i:02d}: pred={labels_text[pred]:6s} | probs={np.round(probs,3)}")
    print("=== Ende Random-Tests ===\n")

# Wenn nur Random-Test gewünscht, fahren und exit
if args.random_only:
    test_random_input(args.n_random)
    sys.exit(0)

# --- LSL-Stream finden ---
def wait_for_stream(stream_type, timeout=60, retry=1):
    start = time.time()
    while True:
        streams = resolve_byprop('type', stream_type, timeout=retry)
        if streams:
            return StreamInlet(streams[0])
        if time.time() - start > timeout:
            raise TimeoutError(f"Kein {stream_type}-Stream gefunden.")

print("Warte auf EEG-Stream…")
inlet      = wait_for_stream('EEG', timeout=30)
stream_name= inlet.info().name()

# --- Ring-Buffer & Voting-History ---
ring_buffer   = np.zeros((N_EEG_CHANNELS, WINDOW_SIZE), dtype=float)
vote_history  = []
current_marker= None

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BCI Live Feedback")
f_small= pygame.font.SysFont("Arial",18)
f_med  = pygame.font.SysFont("Arial",24)
f_big  = pygame.font.SysFont("Arial",48,bold=True)
f_mark = pygame.font.SysFont("Arial",30,italic=True)
clock  = pygame.time.Clock()

# --- Haupt-Schleife ---
running = True
while running:
    screen.fill((255,255,255))
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running=False

    # Stream-Name
    screen.blit(f_small.render(f"Stream: {stream_name}",True,(0,0,0)),(10,10))

    # Neue Probe
    sample_full,_ = inlet.pull_sample(timeout=0.0)
    if sample_full is not None:
        mcode = int(sample_full[-1])
        if mcode>0:
            current_marker=labels_text[mcode-1]
            vote_history.clear()
        ring_buffer=np.roll(ring_buffer,-1,axis=1)
        ring_buffer[:,-1]=sample_full[:N_EEG_CHANNELS]

    display_probs=None
    current_pred=None
    if not np.any(ring_buffer==0):
        # Vorverarbeitung
        filtered     = lfilter(b, a, ring_buffer, axis=1)
        standardized = exponential_moving_standardize(
                           filtered.T, factor_new=1e-3, init_block_size=100
                       ).T
        x = torch.from_numpy(standardized[None]).to(device).float()
        with torch.no_grad():
            logits_all = model(x)
            probs_all  = torch.softmax(logits_all, dim=1)
            probs      = probs_all.mean(dim=2).cpu().numpy().ravel()
        print("Live-Probs:", np.round(probs,3))

        # Voting
        display_probs = probs
        pi = int(probs.argmax())
        vote_history.append(pi)
        if len(vote_history)>VOTE_HISTORY_LENGTH:
            vote_history.pop(0)
        current_pred = max(set(vote_history), key=vote_history.count)

    # Marker-Label
    if current_marker:
        txt = f_mark.render(f"Marker: {current_marker}", True, (50,50,50))
        screen.blit(txt, (SCREEN_WIDTH//2 - txt.get_width()//2,10))
    # Prediction-Label
    if current_pred is not None:
        pr = f_big.render(labels_text[current_pred].upper(), True, colors[current_pred])
        screen.blit(pr, pr.get_rect(center=(SCREEN_WIDTH//2,80)))
    # Wahrscheinlichkeitsbalken
    for i, cls in enumerate(labels_text):
        x = i*(SCREEN_WIDTH//len(labels_text))+50
        screen.blit(f_med.render(cls,True,(0,0,0)),(x,SCREEN_HEIGHT-30))
        if display_probs is not None:
            h = int(display_probs[i]*(SCREEN_HEIGHT-140))
            pygame.draw.rect(screen, colors[i], (x, SCREEN_HEIGHT-60-h, 50, h))
            screen.blit(f_med.render(f"{display_probs[i]:.2f}",True,(0,0,0)),
                        (x, SCREEN_HEIGHT-60-h-25))
        else:
            pygame.draw.rect(screen,(200,200,200),(x,SCREEN_HEIGHT-60-50,50,50))

    pygame.display.flip()
    clock.tick(125)

pygame.quit()
