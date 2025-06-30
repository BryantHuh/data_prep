# 📂 Stream

Skripte zur Verarbeitung und Bewertung von EEG-Daten im Live-Stream via LabStreamingLayer (LSL), inkl. GUI-Komponenten zur Visualisierung.

## Inhalte

- `bias_test.py`: Prüft, ob ein Modell ein Bias für eine bestimmte Klasse zeigt.
- `classify_stream.py`: Verifiziert die gestreamten Daten.
- `gui.py`: GUI-Anwendung zur Anzeige von Live-Predictions eines geladenen Modells.
- `streamer_dummy.py`: Dummy-Skript zum Senden einer Datei per LSL.
- `stream_moabb_raw.py`: Streamt rohe MOABB-Daten.
- `stream_moabb_subject.py`: Stream eines ausgewählten MOABB-Subjekts (nicht best-practice).
- `stream_moab_with_marker.py`: Fügt Marker zu gestreamten MOABB-Daten hinzu.
- `stream_xdf.py`: Streamt gespeicherte `.xdf`-Dateien per LSL.
- `test_moabb.py`: Gibt die Klassenverteilung der MOABB-Fenster aus.
- `test_xdf.py`: Streamt `.xdf`-Dateien und analysiert Debug-Ausgaben.
