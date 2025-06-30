# 📂 Data Compare

Vergleich und Analyse zweier Datenzugänge zum BCI Competition IV 2a Datensatz:
- a) MOABB-Version (vorgefenstert, standardisiert)
- b) Lokale GDF-Dateien (originale Rohdaten)

## Inhalte

- `compare.py`: Vergleich mit Plots (derzeit inkorrekt angewendet, da MOABB bereits vorgeschnitten).
- `compare_one_on_one.py`: Vergleicht Shapes zwischen lokaler GDF und MOABB.
- `compare_trials.py`: Vergleich von Anzahl und Inhalten einzelner Trials.
- `compare_trial_data_all_runs.py`: Vergleich aller Runs zwischen MOABB und GDF.
- `compare_trial_data_run0.py`: Vergleich des ersten Runs im Detail.
- `data_exploration_gdf.py`: Grundlagenanalyse zum GDF-Format.
- `data_exploration_gdf_relabel_csv.py`: Export von GDF-Daten in CSV.
- `data_exploration_gdf_relabel_plot.py`: Visualisierung der Labels in GDF-Daten.
- `data_exploration_moabb.py`: Exploration des MOABB-Datensatzes.
- `gs_model_on_offline_bciIV.py`: Versuch, das "Good Subjects" Modell auf GDF-Daten anzuwenden.
- `show_moabb.py`: Anzeige der Struktur des MOABB-Datensatzes.
- `subject1_model_create_local.py`: Training eines Modells auf Subjekt 1 (lokale GDF).
