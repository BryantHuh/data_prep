# Modell-Evaluierung

Dieses Verzeichnis enthält umfassende Evaluierungswerkzeuge zur Bewertung der BCI-Modellleistung. Das Evaluierungssystem bietet eine detaillierte Analyse der Modellgenauigkeit, Konfusionsmatrizen und Leistungsmetriken.

## 🎯 Evaluierungsbegründung

### Warum umfassende Evaluierung?
- **Modellvergleich**: Vergleich der Leistung von EEGNetv4 und ShallowFBCSPNet
- **Leistungsvalidierung**: Sicherstellen, dass Modelle Echtzeitanforderungen erfüllen
- **Qualitätssicherung**: Potenzielle Probleme vor dem Einsatz identifizieren
- **Forschungs-Insights**: Modellverhalten und -grenzen verstehen

### Evaluierungsstrategie
- **Mehrere Metriken**: Genauigkeit, Konfusionsmatrizen, Klassifikationsberichte
- **Visuelle Analyse**: Plots und Visualisierungen für besseres Verständnis
- **Statistische Strenge**: Korrekte Evaluierungsmethodik
- **Praxisrelevanz**: Fokus auf praktische BCI-Anwendungen

## 📁 Verfügbare Skripte

### `evaluate_models.py`
- **Zweck**: Umfassende Modellevaluierung mit mehreren Metriken
- **Verwendung**: `python evaluate_models.py`
- **Features**:
  - Genauigkeitsberechnung
  - Erstellung von Konfusionsmatrizen
  - Klassifikationsberichte
  - Leistungsvisualisierung
- **Ausgabe**: Ergebnisse werden im `logs/`-Verzeichnis gespeichert

## 🔧 Evaluierungsprozess

### Standard-Evaluierungspipeline
1. **Modell laden**: Geladene Modelle aus dem `models/`-Verzeichnis
2. **Datenvorbereitung**: Testdatensätze vorbereiten
3. **Vorhersagegenerierung**: Modelle auf Testdaten anwenden
4. **Metrikberechnung**: Genauigkeit, Präzision, Recall, F1-Score berechnen
5. **Visualisierung**: Konfusionsmatrizen und Leistungsplots erzeugen
6. **Berichtserstellung**: Detaillierte Evaluierungsberichte speichern

### Wichtige Metriken
- **Genauigkeit**: Gesamtklassifikationsleistung
- **Präzision**: Wahre Positive / (Wahre Positive + Falsch Positive)
- **Recall**: Wahre Positive / (Wahre Positive + Falsch Negative)
- **F1-Score**: Harmonisches Mittel von Präzision und Recall
- **Konfusionsmatrix**: Klassenspezifische Leistungsanalyse

## 📊 Leistungsanalyse

### Erwartete Leistungsbereiche
**Hinweis**: Diese Bereiche variieren aufgrund von ML-Zufälligkeit und dienen als Richtwerte, nicht als Garantien.

- **EEGNetv4**: 70-85% Genauigkeit
- **ShallowFBCSPNet**: 65-80% Genauigkeit
- **Multi-Subjekt-Modelle**: Können verbesserte Generalisierung zeigen

### Klassenspezifische Leistung
- **Fuß-Imagery**: Oft höchste Genauigkeit
- **Hand-Imagery**: Mittlere Leistung
- **Zungen-Imagery**: Kann für einige Subjekte herausfordernd sein

## 🎯 Entscheidungsbegründungen

### Warum dieser Evaluierungsansatz?
1. **Umfassende Metriken**: Mehrere Perspektiven auf die Modellleistung
2. **Visuelle Analyse**: Leichter verständlich als reine Zahlen
3. **Standardmethodik**: Entspricht BCI-Forschungskonventionen
4. **Praxisfokus**: Betonung der realen Anwendbarkeit

### Warum diese Metriken?
- **Genauigkeit**: Primärer Leistungsindikator für BCI-Anwendungen
- **Konfusionsmatrix**: Zeigt klassenspezifische Stärken und Schwächen
- **Präzision/Recall**: Wichtig für Verständnis von Fehlerraten
- **F1-Score**: Ausgewogene Metrik für unausgeglichene Datensätze

### Warum dieses Framework?
- **Braindecode-Integration**: Nahtlose Evaluierung mit Trainings-Framework
- **Standardtools**: Verwendung etablierter Evaluierungsbibliotheken
- **Reproduzierbarkeit**: Konsistente Evaluierungsmethodik
- **Erweiterbarkeit**: Leicht um neue Metriken oder Modelle erweiterbar

## 📈 Ergebnisse verstehen

### Genauigkeitsvariabilität
Die Leistung von Machine-Learning-Modellen variiert zwischen Läufen aufgrund von:
- **Zufälliger Initialisierung**: Unterschiedliche Startgewichte
- **Stochastische Optimierung**: Zufallselemente im Training
- **Daten-Shuffling**: Unterschiedliche Trainings-/Validierungsaufteilungen

Dies ist normales Verhalten und kein Hinweis auf Systemprobleme.

### Konfusionsmatrizen interpretieren
- **Diagonalelemente**: Korrekte Klassifikationen
- **Nebendiagonalelemente**: Fehlklassifikationen
- **Zeilensummen**: Gesamte Stichproben pro wahrer Klasse
- **Spaltensummen**: Gesamte Vorhersagen pro vorhergesagter Klasse

### Leistungsbenchmarks
- **Exzellent**: >80% Genauigkeit
- **Gut**: 70-80% Genauigkeit
- **Akzeptabel**: 60-70% Genauigkeit
- **Verbesserungswürdig**: <60% Genauigkeit

## 🚀 Schnellstart

1. **Modelle zuerst trainieren**: Sicherstellen, dass Modelle im `models/`-Verzeichnis existieren
2. **Evaluierung ausführen**: `python evaluate_models.py`
3. **Ergebnisse prüfen**: Plots und Berichte im `logs/`-Verzeichnis ansehen
4. **Leistung analysieren**: Konfusionsmatrizen für detaillierte Analyse nutzen

## 🔄 Framework-Überlegungen

### Braindecode-Einschränkungen
- **Begrenzte Evaluierungstools**: Nur grundlegende Metriken
- **Dokumentationslücken**: Evaluierungsverfahren schlecht dokumentiert
- **Anpassung**: Begrenzte Flexibilität für eigene Metriken

### Zukünftige Verbesserungen
- **Erweiterte Metriken**: Komplexere Evaluierungsmaßnahmen hinzufügen
- **Cross-Validation**: Korrekte Cross-Validation implementieren
- **Statistische Tests**: Signifikanztests für Modellvergleiche
- **Echtzeit-Evaluierung**: Streaming-Leistungsmetriken evaluieren

## 📝 Ausgabedateien

### Generierte Dateien
- **Konfusionsmatrizen**: Visuelle klassenspezifische Leistung
- **Leistungsplots**: Trainings- und Validierungskurven
- **Evaluierungsberichte**: Detaillierte numerische Ergebnisse
- **Logdateien**: Vollständige Evaluierungsprozess-Logs

### Dateipfade
- **Plots**: `logs/evaluation_*.png`
- **Berichte**: `logs/evaluation_*.csv`
- **Logs**: `logs/evaluation_*.log`

## 🎯 Best Practices

### Evaluierungsrichtlinien
1. **Nur Testdaten verwenden**: Niemals auf Trainingsdaten evaluieren
2. **Mehrere Läufe**: Evaluierung mehrfach durchführen
3. **Statistische Signifikanz**: Konfidenzintervalle berücksichtigen
4. **Praxisbezug**: Fokus auf praktische BCI-Anwendungen

### Häufige Fehlerquellen
- **Overfitting**: Modelle performen gut im Training, schlecht im Test
- **Data Leakage**: Testdaten versehentlich im Training verwendet
- **Zu wenig Daten**: Zu wenige Stichproben für verlässliche Evaluierung
- **Klassenungleichgewicht**: Ungleiche Klassenverteilung beeinflusst Metriken

## 🔮 Zukünftige Erweiterungen

### Geplante Verbesserungen
- **Echtzeit-Evaluierung**: Metriken für Streaming-Performance
- **Cross-Subject-Evaluierung**: Generalisierung über Subjekte testen
- **Erweiterte Metriken**: ROC-Kurven, AUC und weitere
- **Automatisierte Berichte**: Umfassende Evaluierungsberichte generieren
- **Leistungstracking**: Historische Leistungsüberwachung