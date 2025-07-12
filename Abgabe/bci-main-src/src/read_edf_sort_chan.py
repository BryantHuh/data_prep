import mne

# einlesen edf file
raw = mne.io.read_raw_edf('data/MI_Movement/S001/S001R01.edf', preload=True)

all_chs = raw.info['ch_names']
print("Alle Kanäle im EDF:", all_chs)
# Nur die Channels die auch wir erstellen können
wanted_new = [
    'Fp1.', 'Fp2.', 'F3..', 'F4..', 'F7..', 'F8..',
    'C3..', 'C4..', 'T7..', 'T8..', 'P3..', 'P4..',
    'P7..', 'P8..', 'O1..', 'O2..'
]
raw.pick_channels(wanted_new)

# Dictionary um die Channels umzubenennen wie sie bei uns sind
mapping = {
    'Fp1.': 'Fp1',
    'Fp2.': 'Fp2',
    'F3..': 'F3',
    'F4..': 'F4',
    'F7..': 'F7',
    'F8..': 'F8',
    'C3..': 'C3',
    'C4..': 'C4',
    'T7..': 'T3',   # alt
    'T8..': 'T4',   # alt
    'P7..': 'T5',   # alt
    'P8..': 'T6',   # alt
    'P3..': 'P3',
    'P4..': 'P4',
    'O1..': 'O1',
    'O2..': 'O2'
}
raw.rename_channels(mapping)

# sortieren der Channels wie in der GUI
reorder_list = ['Fp1','Fp2', 'C3','C4','T5','T6','O1','O2','F7','F8','F3','F4','T3','T4','P3','P4']
raw.reorder_channels(reorder_list)


print(raw.info['ch_names'])