# Trainings-Skripte

Dieses Verzeichnis enthält alle Modell-Trainingsskripte für das BCI-System. Wir konzentrieren uns auf zwei Hauptarchitekturen: EEGNetv4 (primäre Wahl) und ShallowFBCSPNet (Alternative).

## 🎯 Modellauswahl-Begründung

### EEGNetv4 (Primäre Wahl)
**Warum gewählt**: EEGNetv4 wurde als unser primäres Modell aufgrund seiner robusten Performance in Echtzeit-Anwendungen ausgewählt. Es zeigt ausgezeichnete Streaming-Kompatibilität und konsistente Genauigkeit über verschiedene Subjekte hinweg.

**Performance**: Erreicht typischerweise 70-85% Genauigkeit (variiert pro Training aufgrund ML-Zufälligkeit)
**Streaming-Kompatibilität**: Ausgezeichnete Echtzeit-Performance mit minimaler Latenz
**Status**: Derzeit bevorzugt für Echtzeit-Klassifikation

### ShallowFBCSPNet (Alternative)
**Warum gewählt**: Basierend auf Schirrmeister et al. (2017) zeigt dieses Modell gute Offline-Performance und repräsentiert einen gut etablierten Ansatz in der BCI-Literatur.

**Performance**: 65-80% Genauigkeit (variiert pro Training aufgrund ML-Zufälligkeit)
**Streaming-Kompatibilität**: Hat Komplikationen mit Echtzeit-Genauigkeit, die untersucht werden müssen
**Status**: Enthalten für Vollständigkeit, benötigt zukünftige Untersuchung für Streaming-Anwendungen

## 📁 Verfügbare Skripte

### Kern-Trainingsskripte

#### `train_eegnet.py`
- **Zweck**: Trainiert EEGNetv4 auf Einzel-Subjekt (Subjekt 3)
- **Verwendung**: `python train_eegnet.py`
- **Ausgabe**: `models/eegnetv4_subj3_model_250.pth`
- **Begründung**: Einzel-Subjekt-Training bietet Baseline-Performance und schnellere Trainingszeiten

#### `train_eegnet_xdf.py`
- **Zweck**: Trainiert EEGNetv4 auf benutzerdefinierten XDF-Aufnahmen von OpenBCI
- **Verwendung**: `python train_eegnet_xdf.py --xdf-path path/to/recording.xdf`
- **Features**: Unterstützt sowohl Marker-basiertes als auch kontinuierliches Training
- **Begründung**: Ermöglicht personalisierte Modelle aus benutzerdefinierten Aufnahmen

#### `train_shallow_fbcsp.py`
- **Zweck**: Trainiert ShallowFBCSPNet auf Einzel-Subjekt
- **Verwendung**: `python train_shallow_fbcsp.py`
- **Ausgabe**: `models/shallow_fbcsp_model_250.pth`
- **Begründung**: Bietet Vergleichs-Baseline für alternative Architektur

### Erweiterte Trainingsskripte

#### `train_shallow_fbcsp_good_subjects.py`
- **Zweck**: Trainiert ShallowFBCSPNet auf allen 4 guten Subjekten (1, 3, 8, 9)
- **Verwendung**: `python train_shallow_fbcsp_good_subjects.py`
- **Ausgabe**: `models/shallow_fbcsp_good_subjects_model_250.pth`
- **Begründung**: Multi-Subjekt-Training kann Generalisierung im Vergleich zu Einzel-Subjekt-Modellen verbessern

#### `train_shallow_fbcsp_leave_one_out.py`
- **Zweck**: Leave-one-out Cross-Validation mit guten Subjekten
- **Verwendung**: `python train_shallow_fbcsp_leave_one_out.py`
- **Ausgabe**: Mehrere Modelle für jede Subjekt-Kombination
- **Begründung**: Robuste Evaluierung der Modell-Generalisierung über Subjekte hinweg

## 🔧 Trainings-Konfiguration

### Gemeinsame Parameter
- **Fenstergröße**: 250 Samples (2 Sekunden bei 125 Hz)
- **Kanäle**: 16 Standard-EEG-Kanäle
- **Klassen**: 4 (Füße, linke Hand, rechte Hand, Zunge)
- **Batch-Größe**: 32
- **Epochen**: 100 mit Early Stopping
- **Lernrate**: 0.001 mit Cosine Annealing

### Preprocessing-Pipeline
1. **Kanalauswahl**: 16 Standard-Kanäle
2. **Skalierung**: Konvertierung zu Mikrovolt
3. **Resampling**: 125 Hz
4. **Fenster-Erstellung**: 2-Sekunden-Fenster mit 0.5s Offset

## 📊 Performance-Erwartungen

**Wichtiger Hinweis**: Die Genauigkeit von Machine Learning Modellen variiert zwischen Trainingsläufen aufgrund von:
- Zufälliger Gewichtsinitialisierung
- Stochastischen Optimierungsprozessen
- Daten-Shuffling während des Trainings

Dies ist normales Verhalten im Deep Learning und nicht indikativ für Systemprobleme.

### Typische Performance-Bereiche
- **EEGNetv4**: 70-85% Genauigkeit
- **ShallowFBCSPNet**: 65-80% Genauigkeit
- **Multi-Subjekt-Modelle**: Können verbesserte Generalisierung zeigen

## 🚀 Schnellstart

1. **Abhängigkeiten installieren**: `pip install -r requirements.txt`
2. **EEGNetv4 trainieren**: `python train_eegnet.py`
3. **Ergebnisse evaluieren**: Überprüfen Sie `logs/` Verzeichnis für Plots
4. **Für Streaming verwenden**: Modell gespeichert in `models/` Verzeichnis

## 🔄 Framework-Überlegungen

### Braindecode-Einschränkungen
- **Dokumentation**: Begrenzte und veraltete Dokumentation
- **Community**: Kleinere Benutzerbasis
- **Wartung**: Nicht aktiv gewartet

### Zukünftige Empfehlungen
Erwägen Sie alternative Frameworks für zukünftige Entwicklung:
- **PyTorch Lightning**: Bessere Dokumentation und aktive Community
- **TensorFlow/Keras**: Umfangreiches Ökosystem und Tutorials
- **scikit-learn**: Einfacherer Ansatz für Baseline-Modelle

## 📈 Trainings-Überwachung

### Logs und Ausgaben
- **Trainings-Plots**: Gespeichert in `logs/` Verzeichnis
- **Confusion Matrices**: Visuelle Performance-Analyse
- **Modell-Checkpoints**: Gespeichert in `models/` Verzeichnis
- **Konsolen-Ausgabe**: Echtzeit-Trainingsfortschritt

### Wichtige Metriken
- **Trainings-/Validierungs-Loss**: Überwachung von Overfitting
- **Genauigkeit**: Primäre Performance-Metrik
- **Confusion Matrix**: Klass-spezifische Performance
- **Lernkurven**: Trainingsstabilität

## 🎯 Entscheidungsbegründungen

### Warum diese Modelle?
1. **EEGNetv4**: Bewiesene Echtzeit-Performance, einfache Architektur
2. **ShallowFBCSPNet**: Etablierte Literatur, gute Offline-Performance
3. **Multi-Subjekt-Training**: Potenzial für bessere Generalisierung
4. **Leave-one-out**: Robuste Evaluierungsmethodik

### Warum dieses Preprocessing?
- **Standard-Kanäle**: Industriestandard-Elektrodenauswahl
- **125 Hz Sampling**: Balance zwischen Performance und Rechenkosten
- **2-Sekunden-Fenster**: Optimal für motorische Vorstellungs-Klassifikation
- **Framework-basiert**: Schnelle Implementierung mit Braindecode

### Zukünftige Verbesserungen
- Erforschen Sie alternative Preprocessing-Techniken
- Untersuchen Sie Streaming-Genauigkeitsprobleme mit ShallowFBCSPNet
- Erwägen Sie Ensemble-Methoden für verbesserte Performance
- Implementieren Sie Cross-Validation für alle Modelle