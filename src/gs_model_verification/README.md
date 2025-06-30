# 📂 GS Model Verification

Skripte zur Validierung der Generalisierungsfähigkeit des "Good Subjects"-Modells (Subjekte 1, 3, 8 und 9).

## Inhalte

- `all_trials_subj8.py`: Genauigkeitsprüfung des Modells auf allen Trials von Subjekt 8.
- `all_windows.py`: Genauigkeitsprüfung auf Fenster-Ebene von Subjekt 8.
- `check_train_and_test_same_data.py`: Validiert das Modell auf den Trainingsdaten zur Überprüfung der Trainingskonsistenz.
- `loso.py`: Leave-One-Subject-Out-Test mit Good-Subjects.
- `multi_trial_prediction_test.py`: Weitere Analyse auf Trial-Ebene bei Subjekt 8.
- `one_trial_input_test.py`: Einzel-Input-Test mit Groundtruth-Vergleich.
- `one_window_test.py`: Analyse auf Fensterbasis.
- `sessions.py`: Evaluation der Modellleistung in verschiedenen Sessions.
