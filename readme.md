# 🧠 EEG-Gaming Projekt – BCI mit Braindecode & MOABB

Dieses Repository dokumentiert ein studentisches Projekt zur Steuerung eines Spiels mit EEG-Daten. Es basiert auf Modellen, die mithilfe von Braindecode (Deep Learning für EEG) und der MOABB-Datenbank (Mother Of All BCI Benchmarks) trainiert wurden.

## 🎯 Ziel

- Umsetzung der Forschungsfrage:
  **„Wie kann man am einfachsten ein EEG benutzen, um einen Computer zu steuern?“**

- Aufbau eines EEG-Klassifikationsmodells, das motorische Imaginationen (Links/Rechts) erkennt und in Interaktionen übersetzt.

## 🧱 Projektstruktur

| Ordner             | Inhalt |
|--------------------|--------|
| [`src/`](src/README.md) | Sämtlicher Quellcode: Modelltraining, Tests, Stream-Auswertung |
| [`models/`](models/README.md) | Alle trainierten Modelle (pro Subjekt, „Good Subjects“, etc.) |
| [`log/`](log/README.md) | Visualisierungen, Konfusionsmatrizen, Vergleichsplots |
| [`results/`](results) | Aggregierte Ergebnisse, z. B. Accuracy pro Subjekt (CSV) |
| [`data/`](data) | Rohdaten, vorbereitete Formate, Annotationen |
| [`requirements.txt`](requirements.txt) | Python-Abhängigkeiten für Training, Analyse und Streaming |

## 📌 Verzeichnisse im Detail

- 📂 [`src/braindecode_way`](src/braindecode_way/README.md): Modelle nach dem Braindecode-Tutorialansatz
- 📂 [`src/change_input_windows`](src/change_input_windows/README.md): Analyse verschiedener Fenstergrößen
- 📂 [`src/data_compare`](src/data_compare/README.md): Vergleich MOABB vs. GDF
- 📂 [`src/gs_model_verification`](src/gs_model_verification/README.md): Tests zur Generalisierung des „Good Subjects“-Modells
- 📂 [`src/load_model`](src/load_model/README.md): Lade- und Testskripte für gespeicherte Modelle
- 📂 [`src/stream`](src/stream/README.md): Auswertung gestreamter Daten mit GUI und LSL

## 👥 Beteiligte

- Entwicklungsteam: Thorben, Alex & Tilon als PA2 Auftrag von Prof. Felix Woelk
- Basis: Studie von Schirrmeister et al. (2017), Braindecode, MOABB

## 🛠️ Setup

```bash
pip install -r requirements.txt
