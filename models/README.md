# 📂 Models
In diesem Verzeichnis befinden sich die Modelle, die im Rahmen des Projekts als relevant eingestuft wurden.

## Struktur und Konvention
Modelle mit full im Dateinamen enthalten das komplette Modellobjekt, inklusive Architektur und Gewichtungen.

Modelle ohne full enthalten lediglich die trainierten Gewichtungsparameter (state_dict), was für das Nachladen ausreichend ist.

## Enthaltene Modelltypen
### 🔢 Subjektmodelle: Für jedes Subjekt des BCI Competition IV 2a-Datensatzes (Subjekte 1 bis 9) wurde ein separates Modell trainiert.

### ⏱️ Fenstergrößenvarianten: Es wurden Modelle mit unterschiedlichen Input-Fenstergrößen (z. B. 100 bis 500 Samples) erstellt, um die Sensitivität gegenüber zeitlicher Auflösung zu analysieren.

### ⭐ „Good Subjects“-Modell: Ein Modell wurde ausschließlich auf den als qualitativ hochwertig eingestuften Subjekten 1, 3, 8 und 9 trainiert.

### 🔄 Leave-One-Out-Modelle: Für Generalisierungstests wurde ein Subjekt jeweils ausgelassen (LOSO-Ansatz), um die Robustheit gegenüber unbekannten Teilnehmern zu evaluieren.

### 🧠 All-Subjects-Modell: Ein zentrales Modell wurde mit den Daten aller Subjekte trainiert, um maximale Variabilität abzudecken.

Diese Modelle dienen als Grundlage für Vergleichsstudien, Generalisierungstests sowie erste Ansätze zur Echtzeitanwendung.