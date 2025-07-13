# Trainierte Modelle

Dieses Verzeichnis enthält alle trainierten BCI-Modelle. Die Modelle werden in verschiedenen Formaten gespeichert und können für Echtzeit-Klassifikation und Evaluierung verwendet werden.

## 🎯 Modell-Übersicht

### EEGNetv4 Modelle (Bevorzugt für Echtzeit-Anwendungen)

#### `eegnetv4_subj3_model_250.pth`
- **Zweck**: Vollständig trainiertes EEGNetv4 Modell für Subjekt 3
- **Verwendung**: Echtzeit-Klassifikation und Evaluierung
- **Performance**: Typischerweise 70-85% Genauigkeit
- **Status**: Bevorzugtes Modell für Echtzeit-Anwendungen

#### `eegnetv4_subj3_model_250_full.pth`
- **Zweck**: Vollständiges Modell mit allen Metadaten
- **Verwendung**: Modell-Analyse und Debugging
- **Inhalt**: Modell + Trainings-Historie + Konfiguration
- **Größe**: Größer als .pth Datei, enthält zusätzliche Informationen

### ShallowFBCSPNet Modelle (Alternative für Offline-Anwendungen)

#### `shallow_fbcsp_subj3_model_250.pth`
- **Zweck**: ShallowFBCSPNet Modell für Subjekt 3
- **Verwendung**: Offline-Klassifikation und Vergleich
- **Performance**: 65-80% Genauigkeit
- **Status**: Gute Offline-Performance, Streaming-Komplikationen

#### `shallow_fbcsp_subj3_model_250_full.pth`
- **Zweck**: Vollständiges ShallowFBCSPNet Modell für Subjekt 3
- **Verwendung**: Detaillierte Modell-Analyse
- **Inhalt**: Modell + Trainings-Historie + Konfiguration

#### `shallow_fbcsp_good_subjects_model_250.pth`
- **Zweck**: Multi-Subjekt Modell (Subjekte 1, 3, 8, 9)
- **Verwendung**: Verbesserte Generalisierung
- **Performance**: Potenziell bessere Generalisierung
- **Status**: Experimentell, für Forschungszwecke

#### `shallow_fbcsp_good_subjects_model_250_full.pth`
- **Zweck**: Vollständiges Multi-Subjekt Modell
- **Verwendung**: Detaillierte Analyse der Multi-Subjekt-Performance
- **Inhalt**: Modell + Trainings-Historie + Konfiguration

### Test-Modelle

#### `test_model.pth`
- **Zweck**: Test-Modell für Entwicklung und Debugging
- **Verwendung**: System-Tests und Validierung
- **Status**: Nur für Entwicklungszwecke

## 🔧 Modell-Verwendung

### Laden eines Modells
```python
import torch
from braindecode.models import EEGNetv4, ShallowFBCSPNet

# EEGNetv4 laden
model = EEGNetv4(n_chans=16, n_outputs=4, n_times=250)
model.load_state_dict(torch.load('models/eegnetv4_subj3_model_250.pth'))

# ShallowFBCSPNet laden
model = ShallowFBCSPNet(n_chans=16, n_outputs=4, n_times=250)
model.load_state_dict(torch.load('models/shallow_fbcsp_subj3_model_250.pth'))
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

## 🎯 Entscheidungsbegründungen

### Warum verschiedene Modell-Formate?
- **.pth Dateien**: Kompakte Speicherung für Produktionsanwendungen
- **Full-Modelle**: Vollständige Informationen für Forschung und Debugging
- **Multi-Subjekt**: Experimentelle Ansätze für bessere Generalisierung

### Warum EEGNetv4 als Primärwahl?
- **Echtzeit-Performance**: Ausgezeichnete Streaming-Kompatibilität
- **Konsistenz**: Stabile Performance über verschiedene Subjekte
- **Einfachheit**: Weniger komplexe Architektur für schnelle Inferenz

### Warum ShallowFBCSPNet als Alternative?
- **Literatur-Standard**: Basierend auf etablierter Forschung
- **Offline-Performance**: Gute Ergebnisse in Offline-Szenarien
- **Vergleichsbasis**: Ermöglicht Modell-Vergleiche

## 🚀 Schnellstart

### Modell für Echtzeit-Klassifikation verwenden
1. **EEGNetv4 laden**: Verwenden Sie `eegnetv4_subj3_model_250.pth`
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
- **Echtzeit-Anwendungen**: Verwenden Sie EEGNetv4
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