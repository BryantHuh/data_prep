# test_offline_pipeline.py
import torch
import numpy as np
from scipy.signal import butter, lfilter
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset

# 1) Lade das Modell genau wie in Deinem GUI
MODEL_PATH = "…/models/moabb_downsampled_good_subjects_model_full.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device).eval()

# 2) Lade einen Valid-Trial, komplett vorverarbeitet wie offline
subject_ids = [1,3,8,9]
dsets = [MOABBDataset("BNCI2014_001",[s]) for s in subject_ids]
ds = BaseConcatDataset(dsets)
# dieselben Preprocessor wie beim Streamer-Script
from braindecode.preprocessing import Preprocessor, preprocess
preprocs = [
    Preprocessor('pick_channels', ch_names=ds.datasets[0].raw.ch_names, ordered=True),
    Preprocessor(lambda x: x*1e6),
    Preprocessor('resample', sfreq=125),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(exponential_moving_standardize, apply_on_array=True,
                 factor_new=1e-3, init_block_size=1000)
]
preprocess(ds, preprocs, n_jobs=1)
# jetzt valid windows so erstellen wie im Training
from braindecode.preprocessing import create_windows_from_events
sfreq = ds.datasets[0].raw.info['sfreq']
trial_start_offset = int(-0.5*sfreq)
input_window = 500
# Dummy stride, wir wollen nur ein Fenster pro Trial
wins = create_windows_from_events(ds,
    trial_start_offset_samples=trial_start_offset,
    trial_stop_offset_samples=0,
    window_size_samples=input_window,
    window_stride_samples=input_window,
    drop_last_window=False,
    preload=True
)
valid = wins.split('session')['1test']
# Nimm das erste Fenster, das hat shape (n_chans, input_window)
x_np, y_true = valid[0]
print("Ground-Truth Label Index:", y_true)
# 3) Führ’s durch Dein Live-Preprocessing (Bandpass+EMA)
b,a = butter(4, [4/(0.5*sfreq), 38/(0.5*sfreq)], btype='band')
filtered = lfilter(b, a, x_np, axis=1)
stded    = exponential_moving_standardize(filtered.T, factor_new=1e-3, init_block_size=100).T
# 4) Inferenz
x = torch.from_numpy(stded[None]).to(device).float()
with torch.no_grad():
    logits = model(x)           # [1, C, T']
    probs  = torch.softmax(logits.mean(2), dim=1).cpu().numpy().squeeze()
pred = int(probs.argmax())
print("Live-Pipeline Pred Index:", pred, " Probs:", np.round(probs,3))
e