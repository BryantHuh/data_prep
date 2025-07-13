# Datenaufzeichnung und Analyse

Dieses Verzeichnis enthält Werkzeuge zur Aufzeichnung und Analyse von BCI-Daten. Diese Skripte ermöglichen die individuelle Datenerfassung mit OpenBCI-Hardware und bieten umfassende Analysemöglichkeiten für XDF-Aufzeichnungen.

## 🎯 Begründung für Aufzeichnung

### Warum eigene Aufzeichnung?
- **Personalisierte Modelle**: Modelle auf individuelle Probanden trainieren
- **Reale Daten**: Daten in echten BCI-Szenarien erfassen
- **Qualitätskontrolle**: Datenqualität überwachen und sicherstellen
- **Forschungsflexibilität**: Eigene Versuchsdesigns umsetzen

### Warum XDF-Format?
- **LSL-Kompatibilität**: Native LSL-Aufzeichnungsformat
- **Mehrere Streams**: Unterstützung für EEG-Daten und Marker
- **Synchronisation**: Automatische Zeitstempel-Synchronisation
- **Analysetools**: Umfangreiches Ökosystem an Analysetools

## 📁 Verfügbare Skripte

### Datenanalyse-Tools

#### `analyze_xdf_file.py`
- **Zweck**: Umfassende Analyse von XDF-Aufzeichnungen
- **Verwendung**: `python analyze_xdf_file.py --xdf-path path/to/recording.xdf`
- **Features**:
  - Lädt und analysiert XDF-Dateien aus OpenBCI-Aufzeichnungen
  - Extrahiert EEG-Daten und Marker-Streams
  - Datenqualitätsbewertung und Statistiken
  - Visualisierung von EEG-Signalen und Markern
  - Detaillierte Analyseberichte
- **Ausgabe**: Analyseberichte und Visualisierungen im `logs/`-Verzeichnis

#### `process_edf_channels.py`
- **Zweck**: EDF-Dateien verarbeiten und Kanalnamen standardisieren
- **Verwendung**: `python process_edf_channels.py --edf-path path/to/file.edf`
- **Features**:
  - Lädt und verarbeitet EDF-Dateien
  - Mappt Kanalnamen auf Standard-BCI-Format
  - Sortiert Kanäle für konsistente Verarbeitung
  - Unterstützt verschiedene EDF-Formatvarianten
  - Detaillierte Kanalinformationsberichte
- **Begründung**: Hilft, EDF-Dateien aus verschiedenen Quellen konsistent zu nutzen

## 🎮 Kompletter Aufzeichnungs-Workflow mit OpenBCI GUI und LabRecorder

### Voraussetzungen
- **OpenBCI-Hardware**: Cyton-Board oder ähnlich
- **OpenBCI GUI**: Neueste Version von der OpenBCI-Website
- **LabRecorder**: LabRecorder-Anwendung installieren
- **LSL**: LSL korrekt auf dem System installieren

### Schritt-für-Schritt-Setup

#### 1. Hardware-Setup
1. **OpenBCI-Board** per USB mit Computer verbinden
2. **Elektroden anbringen** nach 10-20-EEG-Standard
3. **Leitgel auftragen** für gute Signalqualität
4. **Verbindungen prüfen**: Impedanz in OpenBCI GUI kontrollieren

#### 2. OpenBCI GUI-Konfiguration
1. **OpenBCI GUI starten**
2. **Board auswählen** (Cyton, Cyton+Daisy, etc.)
3. **LSL-Streaming konfigurieren**:
   - Einstellungen → LSL
   - "Stream to LSL" aktivieren
   - Stream-Name setzen (z.B. "OpenBCI_EEG")
   - Abtastrate einstellen (typisch 125 Hz oder 250 Hz)
4. **Stream prüfen**: EEG-Daten in der GUI sichtbar?

#### 3. LabRecorder-Setup
1. **LabRecorder starten**
2. **Streams auswählen**:
   - EEG-Stream (z.B. "OpenBCI_EEG")
   - Marker-Stream falls `create_marker_stream.py` genutzt wird
3. **Aufzeichnung konfigurieren**:
   - Aufnahmedauer oder kontinuierlich
   - Ausgabeverzeichnis wählen
   - Dateinamen-Konvention festlegen
4. **Testaufzeichnung**: Kurze Testaufnahme zur Überprüfung

#### 4. Marker-Stream-Setup (optional)
1. **Marker-Stream starten**: `python create_marker_stream.py`
2. **Experiment konfigurieren**: Anzahl Trials und Bedingungen festlegen
3. **Synchronisation prüfen**: Marker-Stream in LabRecorder sichtbar?
4. **Synchronisation testen**: Marker stimmen mit Ereignissen überein?

#### 5. Aufnahmesession
1. **Proband vorbereiten**: Versuchsprotokoll erklären
2. **LabRecorder starten**: Aufzeichnung beginnen
3. **Experiment durchführen**: BCI-Protokoll folgen
4. **Qualität überwachen**: EEG-Signal während Aufnahme prüfen
5. **Aufnahme beenden**: LabRecorder stoppen und XDF speichern

#### 6. Datenanalyse
1. **Aufnahme analysieren**: `python analyze_xdf_file.py --xdf-path recording.xdf`
2. **Qualität prüfen**: Datenqualitätsbericht ansehen
3. **Modell trainieren**: `python train_eegnet_xdf.py --xdf-path recording.xdf`
4. **Leistung evaluieren**: Modell auf Validierungsdaten testen

## 🔧 Best Practices für Aufzeichnung

### Signalqualität
- **Impedanz**: Elektrodenimpedanz unter 50kΩ halten
- **Gel**: Ausreichend Leitgel verwenden
- **Bewegung**: Probandenbewegung minimieren
- **Umgebung**: Elektrische Störungen reduzieren

### Versuchsdesign
- **Klare Instruktionen**: Probanden genaue Aufgaben erklären
- **Übungssessions**: Vor der Aufnahme üben lassen
- **Pausen**: Erholungsphasen zwischen Blöcken
- **Konsistente Zeiten**: Einheitliche Versuchsdauer

### Datenmanagement
- **Backups**: Immer Sicherungskopien anlegen
- **Metadaten**: Versuchsparameter dokumentieren
- **Dateinamen**: Konsistente, beschreibende Namen verwenden
- **Organisation**: Strukturierte Verzeichnisse

## 📊 Analysefunktionen

### XDF-Analyse
- **Multi-Stream-Support**: EEG- und Marker-Streams gleichzeitig analysieren
- **Qualitätsbewertung**: Automatische Signalqualitätsbewertung
- **Visualisierung**: Umfassende Plots
- **Statistik**: Detaillierte Leistungsmetriken
- **Export**: Analyseergebnisse in verschiedenen Formaten speichern

### EDF-Verarbeitung
- **Kanal-Mapping**: Verschiedene EDF-Formate auf Standard bringen
- **Qualitätskontrolle**: Kanal-Konfigurationen validieren
- **Formatkonvertierung**: Daten für BCI-Anwendungen vorbereiten
- **Dokumentation**: Detaillierte Kanalberichte erzeugen

## 🎯 Entscheidungsbegründungen

### Warum OpenBCI-Hardware?
- **Open Source**: Transparente Hardware
- **Preiswert**: Günstiger als kommerzielle Systeme
- **Community**: Aktive Nutzerbasis und Dokumentation
- **LSL-Integration**: Native LSL-Unterstützung

### Warum LabRecorder?
- **LSL-nativ**: Speziell für LSL-Streams entwickelt
- **Mehrere Streams**: EEG und Marker
- **Synchronisation**: Automatische Zeitstempel
- **Plattformübergreifend**: Windows, macOS, Linux

### Warum XDF-Format?
- **LSL-Standard**: Native LSL-Aufzeichnung
- **Reiche Metadaten**: Umfassende Stream-Informationen
- **Analysetools**: Großes Ökosystem
- **Zukunftssicherheit**: Langfristige Formatstabilität

### Warum eigene Aufzeichnung?
- **Personalisierung**: Modelle auf Individuen trainieren
- **Reale Validierung**: Test in echten BCI-Szenarien
- **Qualitätskontrolle**: Datenqualität überwachen
- **Forschungsflexibilität**: Eigene Designs

## 🚀 Schnellstart

### Grundlegendes Aufzeichnungs-Setup
1. **Software installieren**: OpenBCI GUI, LabRecorder, LSL
2. **Hardware verbinden**: OpenBCI-Board und Elektroden
3. **Streams konfigurieren**: LSL-Streaming in OpenBCI GUI
4. **Aufnahme starten**: Mit LabRecorder Daten erfassen
5. **Daten analysieren**: Analyse-Tools verwenden

### Analyse-Workflow
1. **Aufnahme laden**: `python analyze_xdf_file.py --xdf-path recording.xdf`
2. **Qualität prüfen**: Qualitätsbericht ansehen
3. **Daten verarbeiten**: Preprocessing-Tools nutzen
4. **Modell trainieren**: Aufnahme für Modelltraining nutzen
5. **Ergebnisse evaluieren**: Modellleistung testen

## 🔄 Framework-Überlegungen

### Braindecode-Einschränkungen
- **Wenige Aufzeichnungsbeispiele**: Kaum eigene Aufzeichnungen dokumentiert
- **Dokumentationslücken**: Aufzeichnungsprozesse schlecht dokumentiert
- **Formatunterstützung**: Begrenzte Unterstützung für eigene Datenformate

### Zukünftige Verbesserungen
- **Erweitertes Preprocessing**: Echtzeit-Vorverarbeitung während Aufnahme
- **Qualitätsmonitoring**: Live-Signalqualitätsbewertung
- **Automatisierte Analyse**: Automatische Qualitätskontrolle und Analyse
- **Cloud-Integration**: Remote-Aufzeichnung und Analyse

## 📝 Ausgabedateien

### Analyse-Ausgaben
- **Qualitätsberichte**: Detaillierte Datenqualitätsbewertung
- **Visualisierungen**: EEG-Plots und Marker-Alignments
- **Statistiken**: Leistungsmetriken und Zusammenfassungen
- **Exportdateien**: Verarbeitete Daten in verschiedenen Formaten

### Aufzeichnungs-Ausgaben
- **XDF-Dateien**: LSL-Aufzeichnungen mit EEG und Markern
- **Metadaten**: Versuchsparameter und Bedingungen
- **Qualitätslogs**: Signalqualitätsüberwachung
- **Backups**: Redundante Kopien zur Datensicherheit

## 🎯 Troubleshooting

### Häufige Aufzeichnungsprobleme
- **Schlechte Signalqualität**: Elektroden und Gel prüfen
- **LSL-Verbindungsprobleme**: Netzwerk und Firewall prüfen
- **Synchronisationsprobleme**: Zeitstempel abgleichen
- **Dateikorruption**: Integrität der Aufnahme prüfen

### Debugging-Schritte
1. **Hardware testen**: OpenBCI-Board prüfen
2. **LSL-Streams prüfen**: `test_lsl_streams.py` verwenden
3. **Qualität überwachen**: Signal während Aufnahme prüfen
4. **Dateien validieren**: XDF-Laden und Analyse testen

## 🔮 Zukünftige Erweiterungen

### Geplante Verbesserungen
- **Live-Qualitätsmonitoring**: Echtzeit-Signalbewertung
- **Automatisiertes Preprocessing**: Vorverarbeitung während Aufnahme
- **Cloud-Aufzeichnung**: Remote-Datenerfassung
- **Erweiterte Analyse**: ML-basierte Qualitätsbewertung
- **Mobile Aufzeichnung**: Smartphone-basierte Lösungen