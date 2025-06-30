# 📂 Braindecode Way

Skripte zur Erstellung und Evaluation von EEG-Modellen auf Basis der Braindecode-Library (vgl. Schirrmeister et al., 2017).

## Inhalte

- `all_models_eval_all_subjects.py`: Erstellt ein Modell aus allen Subjekten.
- `data_preprocessing.py`: Darstellung des Preprocessings mit Braindecode.
- `goodsubjectstest.py`: Leave-One-Out-Validierung mit den "Good Subjects".
- `moabb_all_subjects_save_models.py`: Erstellt und speichert je Subjekt ein Modell nach Braindecode-Schema.
- `moam_eval.py`: Evaluation des "Model of All Models".
- `one_model_all_subjects_training.py`: Trainiert ein Modell auf allen Subjekten gleichzeitig.
- `our_parameters.py`: Modelltraining auf "Good Subjects" mit eigenen Parametern (16 Kanäle, 125 Hz).
- `plot_bcic_iv_2a_moabb_cropped.ipynb`: Beispiel-Notebook von der Braindecode-Website.
- `plot_bcic_iv_2a_moabb_cropped.py`: Beispielskript von der Braindecode-Website.
- `test_theory_training.py`: Modelltraining für ein bestimmtes Subjekt mit ausführlichem Logging.
- `test_theory_validate.py`: Lädt ein Modell und analysiert Validierung und Metadaten.
- `validate_accuracy.py`: 5-faches Training je Subjekt, Speicherung der Accuracy-Werte für Vergleich mit MOABB-Benchmark.
