# 📂 Change Input Windows

Skripte zum Training von Modellen mit unterschiedlichen Fenstergrößen der EEG-Eingabedaten. Ziel ist es, den Einfluss der Window Size auf die Modellleistung zu evaluieren.

## Inhalte

- `our_parameters_changing_windows.py`: Trainingsskript mit eigenen Parametern (16 Kanäle, 125 Hz), jedoch variabler Window-Größe (z. B. von 500 auf 100 Samples reduziert).

Die Ergebnisse dienen der Einschätzung, wie empfindlich das Modell auf unterschiedliche zeitliche Ausschnitte der EEG-Daten reagiert.
