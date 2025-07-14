# Trainierte Modelle

Dieses Verzeichnis enthält alle trainierten BCI-Modelle. Die Modelle werden in verschiedenen Formaten gespeichert und können für Echtzeit-Klassifikation und Evaluierung verwendet werden.

## 🎯 Modell-Übersicht

### EEGNetv4 Modelle (Bevorzugt für Echtzeit-Anwendungen)

#### `eegnetv4_subj3_model_250.pth`
- **Inhalt**: Nur State Dict (Gewichte und Bias-Werte)
- **Zweck**: Kompakte Speicherung für Produktionsanwendungen
- **Verwendung**: Echtzeit-Klassifikation und Evaluierung
- **Performance**: Typischerweise 70-85% Genauigkeit
- **Status**: Bevorzugtes Modell für Echtzeit-Anwendungen

#### `eegnetv4_subj3_model_250_full.pth`
- **Inhalt**: Vollständiges Modell (State Dict + Modell-Architektur + Metadaten)
- **Zweck**: Komplette Modell-Informationen für Analyse und Debugging
- **Verwendung**: Modell-Analyse, Debugging und Forschung
- **Inhalt**: Gewichte + Architektur + Trainings-Historie + Konfiguration
- **Größe**: Größer als .pth Datei, enthält alle Modell-Informationen

### ShallowFBCSPNet Modelle (Alternative für Offline-Anwendungen)

#### `shallow_fbcsp_subj3_model_250.pth`
- **Inhalt**: Nur State Dict (Gewichte und Bias-Werte)
- **Zweck**: Kompakte Speicherung für Produktionsanwendungen
- **Verwendung**: Offline-Klassifikation und Vergleich
- **Performance**: 65-80% Genauigkeit
- **Status**: Gute Offline-Performance, Streaming-Komplikationen

#### `shallow_fbcsp_subj3_model_250_full.pth`
- **Inhalt**: Vollständiges Modell (State Dict + Modell-Architektur + Metadaten)
- **Zweck**: Komplette Modell-Informationen für Subjekt 3
- **Verwendung**: Detaillierte Modell-Analyse und Forschung
- **Inhalt**: Gewichte + Architektur + Trainings-Historie + Konfiguration

#### `shallow_fbcsp_good_subjects_model_250.pth`
- **Inhalt**: Nur State Dict (Gewichte und Bias-Werte)
- **Zweck**: Kompakte Speicherung für Multi-Subjekt-Modell
- **Verwendung**: Verbesserte Generalisierung
- **Performance**: Potenziell bessere Generalisierung
- **Status**: Experimentell, für Forschungszwecke

#### `shallow_fbcsp_good_subjects_model_250_full.pth`
- **Inhalt**: Vollständiges Modell (State Dict + Modell-Architektur + Metadaten)
- **Zweck**: Komplette Multi-Subjekt-Modell-Informationen
- **Verwendung**: Detaillierte Analyse der Multi-Subjekt-Performance
- **Inhalt**: Gewichte + Architektur + Trainings-Historie + Konfiguration

### Test-Modelle

#### `test_model.pth`
- **Inhalt**: Nur State Dict (Gewichte und Bias-Werte)
- **Zweck**: Test-Modell für Entwicklung und Debugging
- **Verwendung**: System-Tests und Validierung
- **Status**: Nur für Entwicklungszwecke

## 🔧 Modell-Verwendung

### Laden eines Modells (State Dict)
```python
import torch
from braindecode.models import EEGNetv4, ShallowFBCSPNet

# EEGNetv4 laden (nur Gewichte)
model = EEGNetv4(n_chans=16, n_outputs=4, n_times=250)
model.load_state_dict(torch.load('models/eegnetv4_subj3_model_250.pth'))

# ShallowFBCSPNet laden (nur Gewichte)
model = ShallowFBCSPNet(n_chans=16, n_outputs=4, n_times=250)
model.load_state_dict(torch.load('models/shallow_fbcsp_subj3_model_250.pth'))
```

### Laden eines vollständigen Modells
```python
import torch

# Vollständiges Modell laden (Gewichte + Architektur + Metadaten)
full_model = torch.load('models/eegnetv4_subj3_model_250_full.pth')
model = full_model['model']  # Modell-Architektur
model.load_state_dict(full_model['state_dict'])  # Gewichte laden
```

### Modell-Evaluierung
```python
# Modell in Evaluierungsmodus setzen
model.eval()

# Vorhersage machen
with torch.no_grad():
    output = model(input_data)
    predictions = torch.softmax(output, dim=1)
```

## 📊 Modell-Performance

### Erwartete Genauigkeiten
**Hinweis**: Diese Werte variieren aufgrund von ML-Zufälligkeit und sollten als Richtlinien betrachtet werden.

- **EEGNetv4 (Subjekt 3)**: 70-85%
- **ShallowFBCSPNet (Subjekt 3)**: 65-80%
- **Multi-Subjekt Modelle**: Potenziell bessere Generalisierung

### Modell-Vergleich
- **EEGNetv4**: Bessere Echtzeit-Performance, konsistente Genauigkeit
- **ShallowFBCSPNet**: Gute Offline-Performance, Streaming-Komplikationen
- **Multi-Subjekt**: Potenziell bessere Generalisierung, längere Trainingszeit

## 🎯 Datei-Unterschiede

### State Dict Dateien (.pth ohne _full)
- **Inhalt**: Nur Gewichte und Bias-Werte
- **Größe**: Kompakt, minimaler Speicherplatz
- **Verwendung**: Produktionsanwendungen, Echtzeit-Klassifikation
- **Laden**: Erfordert separate Modell-Architektur-Definition

### Vollständige Modell-Dateien (_full.pth)
- **Inhalt**: Gewichte + Modell-Architektur + Trainings-Historie + Konfiguration
- **Größe**: Größer, enthält alle Modell-Informationen
- **Verwendung**: Forschung, Debugging, Modell-Analyse
- **Laden**: Selbstständig, enthält alle notwendigen Informationen

## 🚀 Schnellstart

### Modell für Echtzeit-Klassifikation verwenden
1. **EEGNetv4 laden**: Verwenden Sie `eegnetv4_subj3_model_250.pth` (nur Gewichte)
2. **Streaming starten**: `python src/03_streaming/eegnet_gui_classifier.py`
3. **Performance überwachen**: Überprüfen Sie Genauigkeit und Latenz

### Modell evaluieren
1. **Modell laden**: Verwenden Sie das gewünschte Modell
2. **Evaluierung starten**: `python src/02_evaluation/evaluate_models.py`
3. **Ergebnisse analysieren**: Überprüfen Sie Confusion Matrices und Metriken

## 📝 Modell-Details

### Trainings-Parameter
- **Window-Größe**: 250 Samples (2 Sekunden bei 125 Hz)
- **Kanäle**: 16 Standard-EEG-Kanäle
- **Klassen**: 4 (Füße, linke Hand, rechte Hand, Zunge)
- **Batch-Größe**: 32
- **Epochen**: 100 mit Early Stopping
- **Lernrate**: 0.001 mit Cosine Annealing

### Modell-Architekturen
- **EEGNetv4**: Speziell für EEG optimierte CNN-Architektur
- **ShallowFBCSPNet**: CSP-basierte Architektur nach Schirrmeister et al.

## 🔄 Framework-Überlegungen

### Braindecode-Einschränkungen
- **Modell-Kompatibilität**: Nur mit Braindecode-Framework kompatibel
- **Export-Optionen**: Begrenzte Export-Möglichkeiten
- **Framework-Abhängigkeit**: Abhängig von Braindecode-Version

### Zukünftige Verbesserungen
- **Framework-Unabhängigkeit**: Export in ONNX oder andere Formate
- **Modell-Optimierung**: Quantisierung für bessere Performance
- **Cross-Validation**: Mehrere Modell-Versionen für Robustheit
- **Ensemble-Methoden**: Kombination mehrerer Modelle

## 🎯 Best Practices

### Modell-Verwaltung
- **Backup-Kopien**: Behalten Sie Backup-Kopien aller Modelle
- **Versionierung**: Dokumentieren Sie Modell-Versionen und Parameter
- **Performance-Tracking**: Überwachen Sie Modell-Performance über Zeit
- **Qualitätskontrolle**: Validieren Sie Modelle vor Produktionsverwendung

### Modell-Auswahl
- **Echtzeit-Anwendungen**: Verwenden Sie EEGNetv4 (State Dict)
- **Offline-Analyse**: ShallowFBCSPNet für Vergleich
- **Forschung**: Multi-Subjekt-Modelle für Generalisierung
- **Entwicklung**: Test-Modelle für System-Validierung

## 🔮 Zukünftige Erweiterungen

### Geplante Verbesserungen
- **Modell-Kompression**: Kleinere Modelle für mobile Anwendungen
- **Adaptive Modelle**: Dynamische Anpassung an Benutzer
- **Ensemble-Methoden**: Kombination mehrerer Modell-Ansätze
- **Transfer Learning**: Anpassung an neue Subjekte
- **Real-time Training**: Kontinuierliche Modell-Verbesserung