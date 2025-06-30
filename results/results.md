# 📊 Cross-Validation-Ergebnisse: ShallowFBCSPNet auf BCI Competition IV 2a

Zur Validierung unseres eigenen EEG-Klassifikationsmodells haben wir eine 5-fache Trainingswiederholung pro Subjekt durchgeführt und daraus Mittelwert und Standardabweichung der Klassifikationsgenauigkeit berechnet.

Dies entspricht dem Evaluationsansatz der MOABB-Plattform (siehe [moabb.neurotechx.com](https://moabb.neurotechx.com/)), bei dem Benchmarks ebenfalls auf Subjektebene mit mehrfachen Wiederholungen erstellt werden.

## 🔢 Ergebnisse

| Subjekt | Mittelwert (%) | Standardabweichung (%) |
|---------|----------------|-------------------------|
| 1       | 77.29          | 0.68                    |
| 2       | 48.06          | 2.75                    |
| 3       | 89.72          | 1.02                    |
| 4       | 67.57          | 3.02                    |
| 5       | 55.07          | 0.71                    |
| 6       | 51.39          | 2.17                    |
| 7       | 61.81          | 2.16                    |
| 8       | 78.47          | 0.90                    |
| 9       | 73.96          | 0.73                    |


## 📌 Fazit

Unsere Modelle mit ShallowFBCSPNet erreichen pro Subjekt konsistente Genauigkeiten, die im Rahmen der auf MOABB veröffentlichten Benchmarks liegen.
Besonders hervorzuheben ist Subjekt 3 mit über 89 % Accuracy. Subjekt 2 bestätigt hingegen, wie stark die individuelle EEG-Signalqualität die Modellleistung beeinflusst.

---

