# Brain-Computer Interface (BCI) Projekt

Ein umfassendes Brain-Computer Interface System für Echtzeit-EEG-Klassifikation mit motorischen Vorstellungsaufgaben. Dieses Projekt implementiert sowohl Trainings- als auch Echtzeit-Streaming-Fähigkeiten für BCI-Anwendungen.

## 🧠 Projektübersicht

Dieses BCI-System konzentriert sich auf die Klassifikation motorischer Vorstellungen mittels EEG-Signalen und ist speziell für Echtzeit-Anwendungen konzipiert. Das Projekt umfasst vollständige Trainings-Pipelines, Evaluierungswerkzeuge, Echtzeit-Streaming-Fähigkeiten und Aufnahme-Utilities.

### Hauptmerkmale
- **Echtzeit-EEG-Klassifikation** mit EEGNetv4 (funktionsfähige Fallback-Lösung)
- **Multi-Subjekt-Training** Fähigkeiten (ShallowFBCSPNet für Cross-Subject)
- **LSL-Streaming** Integration für Echtzeit-Anwendungen
- **Umfassende Evaluierungswerkzeuge**
- **Aufnahme-Utilities** für benutzerdefinierte Datensätze
- **GUI-basierte Klassifikation** Interface

## 📁 Projektstruktur

```
├── src/
│   ├── 01_training/          # Modell-Trainingsskripte
│   ├── 02_evaluation/        # Modell-Evaluierung und -Analyse
│   ├── 03_streaming/         # Echtzeit-Streaming-Anwendungen
│   ├── 04_recording/         # Datenaufnahme und -Analysewerkzeuge
│   └── utils/                # Gemeinsame Utilities
├── models/                   # Trainierte Modell-Dateien
├── logs/                     # Trainings-Logs, Visualisierungen und Ergebnisse
└── data/                     # Datensatz-Speicher
```

## 🚀 Schnellstart

### Installation
```bash
pip install -r requirements.txt
```

### Grundlegende Verwendung
1. **Modell trainieren**: `python src/01_training/train_eegnet.py`
2. **Modelle evaluieren**: `python src/02_evaluation/evaluate_models.py`
3. **Echtzeit-Klassifikation**: `python src/03_streaming/eegnet_gui_classifier.py`

## 🎯 Modellauswahl & Performance

### ShallowFBCSPNet (Ursprüngliche Wahl)
- **Warum ursprünglich gewählt**: Bessere Performance und Cross-Subject-Fähigkeiten
- **Genauigkeit**: 65-80% (variiert pro Training aufgrund ML-Zufälligkeit)
- **Cross-Subject-Performance**: Ausgezeichnete Generalisierung über Subjekte
- **Status**: Enthalten für Vollständigkeit, Streaming-Komplikationen führten zu Fallback

### EEGNetv4 (Fallback-Lösung)
- **Warum gewählt**: Fallback aufgrund von Streaming-Komplikationen mit ShallowFBCSPNet
- **Genauigkeit**: Typischerweise 70-85% (variiert pro Training aufgrund ML-Zufälligkeit)
- **Streaming-Kompatibilität**: Ausgezeichnete Echtzeit-Performance
- **Status**: Derzeit verwendet für Echtzeit-Klassifikation, funktioniert zuverlässig

**Hinweis**: Die Genauigkeit von Machine Learning Modellen variiert zwischen Trainingsläufen aufgrund von zufälliger Initialisierung und stochastischer Optimierung. Dies ist normales Verhalten im Deep Learning.

## 🔧 Framework-Wahl: Braindecode

### Warum Braindecode?
- **Anfängliche Wahl**: Einfach und "funktioniert einfach" für schnelles Prototyping
- **Benutzerfreundlichkeit**: Unkomplizierte API für EEG-Klassifikation
- **Schnelle Entwicklung**: Ermöglichte schnelle Projektenwicklung

### Einschränkungen & Zukünftige Empfehlungen
- **Dokumentation**: Begrenzte und veraltete Dokumentation
- **Community**: Kleinere Benutzerbasis im Vergleich zu Alternativen
- **Aktualität**: Nicht aktiv gewartet
- **Zukünftige Empfehlung**: Erwägen Sie PyTorch Lightning, TensorFlow/Keras oder scikit-learn für aktivere Frameworks

## 📊 Datensatz: BCI Competition IV Dataset 2a

### Warum dieser Datensatz?
- **Industriestandard**: Am häufigsten in der BCI-Forschung verwendet
- **Gut dokumentiert**: Umfangreiche Literatur und Benchmarks
- **Realistisch**: Repräsentiert realistische BCI-Szenarien
- **Verfügbarkeit**: Einfach über MOABB zugänglich

## 🔄 Preprocessing-Ansatz

### Aktuelle Strategie
- **Framework-basiert**: Verwendung der eingebauten Preprocessing-Funktionen von Braindecode
- **Begründung**: Wollten schnell ein funktionsfähiges Produkt
- **Zukünftige Überlegung**: Sollten andere Frameworks für potenziell besseres Preprocessing erkunden

### Echtzeit vs Offline
- **Konsistenz**: Gleiche Preprocessing-Pipeline für beide
- **Begründung**: Stellt Modell-Kompatibilität zwischen Training und Inferenz sicher

## 🌐 LSL (Lab Streaming Layer) Integration

### Warum LSL?
- **OpenBCI-Kompatibilität**: OpenBCI GUI implementiert bereits LSL
- **LabRecorder-Integration**: Einfache Aufnahme-Erstellung
- **Industriestandard**: Weit verbreitet in BCI-Forschung
- **Plattformübergreifend**: Funktioniert auf verschiedenen Systemen

## 📝 Detaillierte Dokumentation

- **[Trainingsanleitung](src/01_training/README.md)** - Modell-Trainingsverfahren
- **[Evaluierungsanleitung](src/02_evaluation/README.md)** - Modell-Evaluierung und -Analyse
- **[Streaming-Anleitung](src/03_streaming/README.md)** - Echtzeit-Anwendungen
- **[Aufnahme-Anleitung](src/04_recording/README.md)** - Datenaufnahme und -Analyse

## 🎮 Aufnahmen mit OpenBCI GUI und LabRecorder erstellen

### Setup-Prozess
1. **[OpenBCI GUI](https://openbci.com/downloads?_gl=1*xrq4ad*_gcl_au*MTYyNjYxODU1MC4xNzUyNDg0NTA5*_ga*MTM4NzIwNDc2LjE3NTI0ODQ1MDk.*_ga_HVMLC0ZWWS*czE3NTI0ODQ1MDkkbzEkZzAkdDE3NTI0ODQ1MDkkajYwJGwwJGgw)** und **[LabRecorder]**(https://github.com/labstreaminglayer/App-LabRecorder) installieren
2. **LSL-Streams** in OpenBCI GUI konfigurieren
3. **LabRecorder** starten und Streams auswählen
4. **Session aufnehmen** mit Markern
5. **Aufnahmen analysieren** mit unseren Werkzeugen

### Detaillierte Schritte
Siehe [Aufnahme-Anleitung](src/04_recording/README.md) für vollständige Anweisungen.

## 🤝 Beitragen

Dieses Projekt dient als Grundlage für BCI-Forschung und -Entwicklung. Zukünftige Verbesserungen sollten erwägen:
- Lösung der Streaming-Komplikationen mit ShallowFBCSPNet
- Alternative Frameworks mit besserer Dokumentation
- Erweiterte Preprocessing-Techniken
- Verbesserte Echtzeit-Performance
- Erweiterte Modell-Architekturen


---

**Hinweis**: Dieses Projekt priorisiert funktionsfähige Lösungen über perfekte Implementierungen. Der Fokus lag darauf, schnell ein funktionsfähiges BCI-System zu erstellen und dabei wissenschaftliche Strenge beizubehalten.
