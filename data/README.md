# Datensatz-Speicher

Dieses Verzeichnis enthält alle Datensätze, Aufnahmen und Datenvergleiche des BCI-Systems. Die verschiedenen Unterverzeichnisse organisieren Daten nach Typ, Quelle und Verwendungszweck.

## 🎯 Daten-Übersicht

### MOABB-Datensätze
Diese Verzeichnisse enthalten die standardisierten MOABB-Datensätze für verschiedene Subjekte.

#### `subject1_moabb/`
- **Zweck**: MOABB-formatierte Daten für Subjekt 1
- **Inhalt**: Vorverarbeitete EEG-Daten im MOABB-Format
- **Verwendung**: Training und Evaluierung
- **Format**: FIF-Dateien mit Metadaten

#### `subject1/` bis `subject9/`
- **Zweck**: Individuelle Subjekt-Datensätze
- **Inhalt**: Roh-EEG-Daten und vorverarbeitete Daten
- **Verwendung**: Subjekt-spezifische Analysen
- **Besonderheit**: Verschiedene Datenformate und -qualitäten

### Experimentelle Daten

#### `recordings/`
- **Zweck**: Benutzerdefinierte EEG-Aufnahmen
- **Inhalt**: XDF-Dateien von OpenBCI-Aufnahmen
- **Verwendung**: Personalisierte Modell-Trainings
- **Format**: LSL-kompatible XDF-Dateien

#### `stream/`
- **Zweck**: Streaming-Test-Daten
- **Inhalt**: Daten für Echtzeit-Tests
- **Verwendung**: Streaming-Anwendungs-Tests
- **Besonderheit**: Optimiert für LSL-Streaming

### Analyse-Daten

#### `Compare_datasets/`
- **Zweck**: Datensatz-Vergleiche und -Analysen
- **Inhalt**: Vergleiche zwischen verschiedenen Datenquellen
- **Verwendung**: Qualitätsbewertung und -optimierung
- **Format**: Visualisierungen und Berichte

#### `change_windowsize/`
- **Zweck**: Experimente mit verschiedenen Fenstergrößen
- **Inhalt**: Analysen zur optimalen Window-Größe
- **Verwendung**: Hyperparameter-Optimierung
- **Ergebnisse**: Performance-Vergleiche verschiedener Konfigurationen

#### `downsampled/`
- **Zweck**: Downsampling-Experimente
- **Inhalt**: Daten mit reduzierter Sampling-Rate
- **Verwendung**: Performance-Optimierung
- **Ziel**: Balance zwischen Genauigkeit und Geschwindigkeit

## 📊 Daten-Qualitätsbewertung

### Subjekt-Performance
- **Subjekte 1, 3, 8, 9**: "Gute" Subjekte mit hoher Performance
- **Subjekte 2, 4-7**: Variablere Performance
- **Subjekt-spezifische Unterschiede**: Individuelle EEG-Muster

### Daten-Qualitätsmetriken
- **Signal-zu-Rausch-Verhältnis**: Bewertung der Signalqualität
- **Artefakt-Detektion**: Identifikation von Bewegungsartefakten
- **Kanal-Konsistenz**: Überprüfung der Elektroden-Qualität
- **Temporal-Stabilität**: Bewertung der zeitlichen Stabilität

## 🎯 Entscheidungsbegründungen

### Warum MOABB-Datensätze?
- **Standardisierung**: Industriestandard für BCI-Forschung
- **Qualität**: Hochwertige, validierte Daten
- **Vergleichbarkeit**: Benchmarking mit anderen Studien
- **Verfügbarkeit**: Einfacher Zugang über MOABB-Framework

### Warum Subjekt-spezifische Verzeichnisse?
- **Organisation**: Klare Trennung nach Subjekten
- **Analyse**: Einfache Subjekt-spezifische Untersuchungen
- **Vergleich**: Direkte Subjekt-Vergleiche
- **Debugging**: Isolierung von Subjekt-spezifischen Problemen

### Warum experimentelle Daten?
- **Forschung**: Untersuchung neuer Ansätze
- **Optimierung**: Hyperparameter-Tuning
- **Validierung**: Überprüfung von Hypothesen
- **Innovation**: Entwicklung neuer Methoden

### Warum verschiedene Datenformate?
- **Kompatibilität**: Unterstützung verschiedener Hardware
- **Flexibilität**: Anpassung an verschiedene Anwendungen
- **Standardisierung**: Verwendung etablierter Formate
- **Zukunftssicherheit**: Langfristige Datenverfügbarkeit

### Warum separate Daten- und Ergebnis-Verzeichnisse?
- **Organisation**: Klare Trennung zwischen Rohdaten und Ergebnissen
- **Zugänglichkeit**: Einfacher Zugang zu spezifischen Datentypen
- **Konsistenz**: Einheitliche Datenverwaltung
- **Wartbarkeit**: Zentrale Datenverwaltung

## 🚀 Daten-Verwendung

### Training mit MOABB-Daten
1. **Datensatz laden**: Verwenden Sie MOABB-API
2. **Preprocessing**: Standardisierte Vorverarbeitung
3. **Training starten**: Verwenden Sie Trainings-Skripte
4. **Ergebnisse evaluieren**: Überprüfen Sie Performance

### Benutzerdefinierte Aufnahmen verwenden
1. **XDF-Datei laden**: Verwenden Sie Analyse-Skripte
2. **Qualität bewerten**: Überprüfen Sie Signalqualität
3. **Preprocessing**: Anpassung an Trainings-Pipeline
4. **Modell trainieren**: Personalisierte Modelle

### Daten-Vergleiche durchführen
1. **Datensätze laden**: Verschiedene Datenquellen
2. **Metriken berechnen**: Qualitätsbewertung
3. **Visualisierungen erstellen**: Vergleichs-Plots
4. **Berichte generieren**: Dokumentation der Ergebnisse

### Daten-Analysen durchführen
1. **Datensatz-Vergleiche**: Qualitätsbewertung verschiedener Datenquellen
2. **Performance-Optimierung**: Hyperparameter-Tuning
3. **Qualitätskontrolle**: Automatische Datenbewertung
4. **Trend-Analyse**: Historische Datenqualitäts-Entwicklung

## 📝 Daten-Formate

### MOABB-Format
- **Dateityp**: FIF-Dateien (.fif)
- **Struktur**: MNE-Python kompatibel
- **Metadaten**: Umfassende Annotations-Informationen
- **Qualität**: Hochwertige, validierte Daten

### XDF-Format
- **Dateityp**: XDF-Dateien (.xdf)
- **Struktur**: LSL-kompatibel
- **Streams**: EEG-Daten und Marker
- **Synchronisation**: Automatische Zeitstempel-Synchronisation

### EDF-Format
- **Dateityp**: EDF-Dateien (.edf)
- **Struktur**: Standard-EEG-Format
- **Kompatibilität**: Weit verbreitet in EEG-Forschung
- **Verarbeitung**: Benötigt spezielle Konvertierung



## 🔄 Framework-Überlegungen

### Braindecode-Daten-Einschränkungen
- **Format-Beschränkungen**: Begrenzte Unterstützung für benutzerdefinierte Formate
- **Preprocessing**: Framework-spezifische Vorverarbeitung
- **Skalierbarkeit**: Begrenzte Unterstützung für große Datensätze

### Zukünftige Verbesserungen
- **Erweiterte Formate**: Unterstützung für mehr Datenformate
- **Automatische Qualitätskontrolle**: KI-basierte Datenbewertung
- **Cloud-Integration**: Remote-Datenverarbeitung
- **Real-time Preprocessing**: Echtzeit-Datenvorverarbeitung

## 🎯 Best Practices

### Daten-Management
- **Backup-Strategien**: Regelmäßige Sicherung aller Daten
- **Versionierung**: Dokumentation von Datenversionen
- **Qualitätskontrolle**: Automatische Qualitätsbewertung
- **Organisation**: Konsistente Verzeichnisstruktur

### Daten-Analyse
- **Reproduzierbarkeit**: Dokumentation aller Analyseschritte
- **Validierung**: Überprüfung der Ergebnisse
- **Vergleichbarkeit**: Standardisierte Metriken
- **Interpretierbarkeit**: Klare Ergebnis-Darstellung

## 🔮 Zukünftige Erweiterungen

### Geplante Verbesserungen
- **Automatische Datenqualitätsbewertung**: KI-basierte Qualitätskontrolle
- **Erweiterte Preprocessing-Pipeline**: Mehr Vorverarbeitungsoptionen
- **Cloud-basierte Datenverarbeitung**: Skalierbare Datenanalyse
- **Real-time Datenanalyse**: Echtzeit-Datenbewertung
- **Multi-Modal-Daten**: Integration verschiedener Sensordaten