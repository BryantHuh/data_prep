import time
import numpy as np
import torch
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from collections import deque, Counter
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import exponential_moving_standardize

# === Parameter ===
sfreq = 125
window_size = 500
stride = 12
n_channels = 16  # ohne Marker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/moabb_downsampled_good_subjects_model_full.pth"
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]
marker_mapping = {0: "no_event", 1: "feet", 2: "left_hand", 3: "right_hand", 4: "tongue"}

# === Modell laden ===
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()

# === Stream finden ===
print("🔍 Suche EEG-Stream…")
streams = resolve_byprop('type', 'EEG', timeout=5.0)
if not streams:
    print("❌ Kein EEG-Stream gefunden.")
    exit(1)
inlet = StreamInlet(streams[0])
print("✅ Stream verbunden.")

# === Datenpuffer initialisieren ===
eeg_buffer = deque(maxlen=window_size)
marker_buffer = deque(maxlen=window_size)

# === Online-Standardisierung vorbereiten ===
ema_state = None

def preprocess_window(window, ema_state):
    """Wendet exponential_moving_standardize auf Window an (Kanalweise)."""
    import numpy as np

    factor_new = 1e-3
    eps = 1e-4

    if ema_state is None:
        # Initialisierung
        ema_state = {
            'mean': np.mean(window, axis=0, keepdims=True),
            'var': np.var(window, axis=0, keepdims=True)
        }

    # EMA Update
    current_mean = np.mean(window, axis=0, keepdims=True)
    current_var = np.var(window, axis=0, keepdims=True)

    ema_state['mean'] = (1 - factor_new) * ema_state['mean'] + factor_new * current_mean
    ema_state['var'] = (1 - factor_new) * ema_state['var'] + factor_new * current_var

    # Standardisierung
    window_proc = (window - ema_state['mean']) / np.sqrt(ema_state['var'] + eps)

    return window_proc, ema_state

# === Streaming & Inferenz ===
sample_counter = 0

while True:
    sample, _ = inlet.pull_sample(timeout=5.0)
    if sample is None:
        print("⚠️ Kein Sample empfangen.")
        continue

    eeg_sample = sample[:n_channels]
    marker_sample = sample[-1]

    eeg_buffer.append(eeg_sample)
    marker_buffer.append(marker_sample)
    sample_counter += 1

    if sample_counter >= window_size and (sample_counter - window_size) % stride == 0:
        eeg_window = np.array(eeg_buffer).T  # Shape: (n_channels, window_size)
        marker_array = np.array(marker_buffer)

        if np.all(marker_array == 0):
            print("⚠️ Window enthält keine gültigen Marker.")
            continue

        true_label = int(Counter(marker_array).most_common(1)[0][0])

        eeg_window_proc, ema_state = preprocess_window(eeg_window, ema_state)
        x_tensor = torch.tensor(eeg_window_proc, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x_tensor)
            if output.ndim == 3:
                output = output.mean(dim=2)
            pred_label = int(output.argmax(dim=1).item())

        print(f"🧠 Pred: {pred_label} ({marker_mapping[pred_label]}) | "
              f"GT: {true_label} ({marker_mapping.get(true_label, '?')})")
