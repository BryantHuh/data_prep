# Echtzeit-Streaming-Anwendungen

Dieses Verzeichnis enthält alle Echtzeit-BCI-Streaming-Anwendungen. Diese Skripte ermöglichen die Live-EEG-Klassifikation für praktische BCI-Anwendungen unter Verwendung von LSL (Lab Streaming Layer) für die Datenübertragung.

## 🎯 Begründung für Streaming

### Warum Echtzeit-Streaming?
- **Praktische Anwendungen**: Reale BCI-Systeme benötigen Live-Verarbeitung
- **Nutzerfeedback**: Sofortige Klassifikationsergebnisse für Interaktion
- **Forschungsvalidierung**: Modelle unter realistischen Bedingungen testen
- **Entwicklungstests**: Modellleistung im Streaming-Szenario überprüfen

### Warum LSL (Lab Streaming Layer)?
- **OpenBCI-Kompatibilität**: OpenBCI GUI implementiert bereits LSL
- **LabRecorder-Integration**: Einfache Aufzeichnungserstellung für Analyse
- **Industriestandard**: Weit verbreitet in BCI-Forschung und -Anwendungen
- **Plattformübergreifend**: Funktioniert auf verschiedenen Betriebssystemen
- **Mehrere Streams**: Unterstützung für EEG-Daten und Marker gleichzeitig

## 📁 Verfügbare Skripte

### Zentrale Streaming-Anwendungen

#### `eegnet_gui_classifier.py`
- **Zweck**: Haupt-GUI für Echtzeitklassifikation mit EEGNetv4
- **Verwendung**: `python eegnet_gui_classifier.py --model-path models/eegnetv4_subj3_model_250.pth`
- **Features**:
  - Echtzeit-EEG-Klassifikation
  - Optionale Marker-Stream-Unterstützung
  - Dynamische Visualisierung der Konfidenz
  - CSV-Exportfunktion
  - Performance-Monitoring
- **Begründung**: Primäres Interface für Echtzeit-BCI-Experimente

#### `simple_eeg_predictor.py`
- **Zweck**: Leichtgewichtige Vorhersage ohne GUI
- **Verwendung**: `python simple_eeg_predictor.py --model-path models/eegnetv4_subj3_model_250.pth`
- **Features**:
  - Kein GUI-Overhead für schnellere Verarbeitung
  - LSL-Stream-Ein- und -Ausgabe
  - Performance-Monitoring
  - Unterstützung für EEGNetv4 und ShallowFBCSPNet
- **Begründung**: Alternative für Anwendungen mit minimalem Interface

### Daten-Streaming-Tools

#### `stream_moabb_data.py`
- **Zweck**: Streamt MOABB-Datensatz über LSL zum Testen
- **Verwendung**: `python stream_moabb_data.py --subject 3 --duration 60`
- **Features**:
  - Simuliert Echtzeit-EEG-Daten aus aufgezeichneten Datensätzen
  - Inklusive Marker-Streams zur Validierung
  - Konfigurierbare Subjektauswahl und Dauer
- **Begründung**: Ermöglicht Testen von Echtzeitanwendungen mit bekannten Daten

#### `create_marker_stream.py`
- **Zweck**: Marker-Streams für BCI-Experimente erzeugen
- **Verwendung**: `python create_marker_stream.py`
- **Features**:
  - Pygame-Interface zur Experimentsteuerung
  - LSL-Marker-Stream-Ausgabe
  - Konfigurierbare Experimentparameter
  - Visuelles Feedback für den Versuchsablauf
- **Begründung**: Bietet synchronisierte Marker für BCI-Experimente

### Debugging und Testen

#### `test_lsl_streams.py`
- **Zweck**: Testet und listet verfügbare LSL-Streams
- **Verwendung**: `python test_lsl_streams.py`
- **Features**:
  - Listet alle verfügbaren LSL-Streams
  - Testet Verbindung zu spezifischen Streams
  - Test gängiger Stream-Namen und -Typen
- **Begründung**: Essenziell für das Debugging von OpenBCI GUI-Verbindungen

## 🔧 Streaming-Architektur

### LSL-Stream-Struktur
- **EEG-Stream**: Kontinuierliche EEG-Daten von OpenBCI oder simulierten Daten
- **Marker-Stream**: Ereignismarker zur Versuchssynchronisation
- **Vorhersage-Stream**: Klassifikationsergebnisse (optional)

### Echtzeit-Verarbeitungspipeline
1. **Datenaufnahme**: Empfang von EEG-Samples via LSL
2. **Buffer-Management**: Sliding-Window-Verwaltung der EEG-Daten
3. **Preprocessing**: Gleiche Vorverarbeitung wie im Training
4. **Klassifikation**: Modellauswertung auf aktuellem Fenster
5. **Ergebnis-Ausgabe**: Anzeige von Vorhersagen und Konfidenzwerten

## 📊 Performance-Überlegungen

### Latenzanforderungen
- **Ziel-Latenz**: <100ms für Echtzeitanwendungen
- **Fenstergröße**: 2-Sekunden-Fenster (250 Samples bei 125 Hz)
- **Verarbeitungs-Overhead**: Minimale GUI-Updates und Logging

### Genauigkeit vs. Geschwindigkeit
- **EEGNetv4**: Bessere Echtzeit-Performance, konsistente Genauigkeit
- **ShallowFBCSPNet**: Gute Offline-Genauigkeit, Streaming-Komplikationen
- **Fensterüberlappung**: Keine Überlappung für Einfachheit, kann optimiert werden

## 🎯 Entscheidungsbegründungen

### Warum LSL für Streaming?
1. **Industriestandard**: Weit verbreitet in der BCI-Forschung
2. **OpenBCI-Integration**: Nahtlose Kompatibilität mit OpenBCI-Hardware
3. **LabRecorder-Support**: Einfache Aufzeichnungs- und Analyse-Workflows
4. **Plattformübergreifend**: Funktioniert auf Windows, macOS und Linux
5. **Mehrere Streams**: Unterstützung für EEG-Daten und Marker gleichzeitig

### Warum Echtzeitverarbeitung?
- **Praktische Anwendungen**: Reale BCI-Systeme benötigen Live-Verarbeitung
- **Nutzerfeedback**: Sofortige Ergebnisse für Interaktion
- **Forschungsvalidierung**: Modelle unter realistischen Bedingungen testen
- **Entwicklungseffizienz**: Schnellere Iteration und Tests

### Warum GUI vs. Kommandozeile?
- **GUI-Vorteile**: Visuelles Feedback, einfaches Debugging, benutzerfreundlich
- **Kommandozeile-Vorteile**: Geringerer Overhead, schnellere Verarbeitung, Automatisierung
- **Hybrid-Ansatz**: Beide Optionen für verschiedene Anwendungsfälle verfügbar

### Warum 2-Sekunden-Fenster?
- **Optimale Länge**: Balance zwischen Genauigkeit und Reaktionsfähigkeit
- **Motor Imagery**: Ausreichend Zeit für motorische Muster
- **Rechenaufwand**: Angemessene Anforderungen
- **Literaturstandard**: Häufig in der BCI-Forschung verwendet

## 🚀 Schnellstart

### Grundlegende Echtzeit-Klassifikation
1. **EEG-Stream starten**: OpenBCI GUI oder simulierte Daten verwenden
2. **Klassifikator ausführen**: `python eegnet_gui_classifier.py`
3. **Ergebnisse überwachen**: Echtzeit-Vorhersagen und Konfidenz anzeigen

### Test-Setup
1. **Streams testen**: `python test_lsl_streams.py`
2. **Daten simulieren**: `python stream_moabb_data.py`
3. **Marker erzeugen**: `python create_marker_stream.py`

## 🔄 Framework-Überlegungen

### Braindecode-Streaming-Einschränkungen
- **Wenige Streaming-Beispiele**: Kaum Echtzeit-Implementierungen dokumentiert
- **Dokumentationslücken**: Streaming-Prozesse schlecht dokumentiert
- **Performance-Optimierung**: Wenig Hinweise zur Echtzeit-Optimierung

### Zukünftige Verbesserungen
- **Erweitertes Preprocessing**: Optimierung für Echtzeit
- **Modelloptimierung**: Quantisierung und Pruning für schnellere Inferenz
- **Multithreading**: Parallele Verarbeitung für bessere Performance
- **Adaptive Fenster**: Dynamische Fenstergrößen je nach Signalqualität

## 📝 Aufzeichnung mit OpenBCI GUI und LabRecorder

### Kompletter Setup-Prozess

#### 1. Benötigte Software installieren
- **OpenBCI GUI**: Von der OpenBCI-Website herunterladen
- **LabRecorder**: LabRecorder-Anwendung installieren
- **LSL**: LSL korrekt installieren

#### 2. OpenBCI GUI konfigurieren
1. **Hardware verbinden**: OpenBCI-Board mit Computer verbinden
2. **OpenBCI GUI starten**: Anwendung öffnen
3. **LSL konfigurieren**: LSL-Streaming in den Einstellungen aktivieren
4. **Streams prüfen**: Überprüfen, ob EEG-Stream sichtbar ist

#### 3. LabRecorder einrichten
1. **LabRecorder starten**: Anwendung öffnen
2. **Streams auswählen**: EEG- und Marker-Streams wählen
3. **Aufzeichnung konfigurieren**: Parameter einstellen
4. **Aufzeichnung starten**: Datenerfassung beginnen

#### 4. BCI-Experiment durchführen
1. **Marker-Stream starten**: `python create_marker_stream.py`
2. **Experiment beginnen**: Protokoll folgen
3. **Session aufzeichnen**: LabRecorder nimmt alle Streams auf
4. **Daten speichern**: Als XDF-Datei für Analyse exportieren

#### 5. Aufzeichnungen analysieren
1. **Analysetools verwenden**: `python analyze_xdf_file.py`
2. **Eigene Modelle trainieren**: `python train_eegnet_xdf.py`
3. **Leistung evaluieren**: Qualität und Marker prüfen

### Best Practices
- **Synchronisation**: Alle Streams synchronisieren
- **Qualitätskontrolle**: EEG-Signalqualität während der Aufnahme überwachen
- **Backups**: Immer Sicherungskopien anlegen
- **Dokumentation**: Versuchsparameter und Bedingungen dokumentieren

## 🎯 Troubleshooting

### Häufige Probleme
- **LSL-Stream nicht gefunden**: Stream-Namen und Netzwerk prüfen
- **Hohe Latenz**: Verarbeitungspipeline optimieren, GUI-Updates reduzieren
- **Schlechte Genauigkeit**: Preprocessing an Trainingspipeline anpassen
- **Verbindungsfehler**: LSL-Installation und Netzwerk prüfen

### Debugging-Schritte
1. **LSL-Streams testen**: `test_lsl_streams.py` verwenden
2. **Modell laden prüfen**: Modelldatei existiert und lädt korrekt
3. **Performance überwachen**: CPU/GPU-Auslastung und Speicher prüfen
4. **Daten validieren**: EEG-Datenformat prüfen

## 🔮 Zukünftige Erweiterungen

### Geplante Verbesserungen
- **Erweiterte GUI**: Bessere Visualisierungsmöglichkeiten
- **Multi-Modell-Support**: Wechsel zwischen Modellen in Echtzeit
- **Adaptive Verarbeitung**: Dynamische Anpassung an Signalqualität
- **Cloud-Integration**: Remote-Monitoring und Analyse
- **Mobile Unterstützung**: Android/iOS-Apps für mobiles BCI