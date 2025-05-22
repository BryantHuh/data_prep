# -------------------------- Dataset-Objekt vorbereiten --------------------------
ds = create_from_mne_raw(raw, trial_start_offset_samples=int(-0.5 * sfreq), trial_stop_offset_samples=0)
dataset = BaseConcatDataset([ds])