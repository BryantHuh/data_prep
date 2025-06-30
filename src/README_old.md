# SRC

In diesem Directory sind alle Skripte und sonstiger Sourcecode.

## Braindecode_way

hierin sind die Skripte, die die Models nach dem Weg des Tutorials von der Braindecode Library erstellen und validieren. Den Hinweis auf diese Library gab die Schirrmeister-Studie.

- src\braindecode_way\all_models_eval_all_subjects.py: Erstellt ein Model aus allen Subjekten
- src\braindecode_way\data_preprocessing.py: Skript in dem das Datapreprocessing von BRaindecode dargestellt ist.
- src\braindecode_way\goodsubjectstest.py: Skript in dem die Leave one out Methode angewendet wird.
- src\braindecode_way\moabb_all_subjects_save_models.py: Skript in dem für jedes Subjekt ein Model erstellt wird. Ganz nach dem BRaindecode Schema.
- src\braindecode_way\moam_eval.py: Model of all Models erstellung und evaluierung.
- src\braindecode_way\one_model_all_subjects_training.py: Skript in dem ein Model auf allen Subjekten trainiert wird.
- src\braindecode_way\our_parameters.py: Skript in dem ein Model mit den "good subjects" erstellt wird, jedoich unsere Parameter angewendet werden beim preprocessing. (125 Hz, 16 Kanäle)
- src\braindecode_way\plot_bcic_iv_2a_moabb_cropped.ipynb: Jupyter Notebook Beispiel heruntergeladen von der BRaindecode Website
- src\braindecode_way\plot_bcic_iv_2a_moabb_cropped.py: Skript Beispiel heruntergeladen von der BRaindecode Website
- src\braindecode_way\test_theory_training.py: Skript welches ein Model nach Subjekt ID erstellt und mehr Informationen über den Datensatz ausgibt.
- src\braindecode_way\test_theory_validate.py: Skript welches ein Model lädt und mehr Informationen über den Prozess ausgibt.
- src\braindecode_way\validate_accuracy.py: Skript welches für alle Subjekte 5 Models erstellt und trianiert, danach die Accuracy in eine CSV plottet, um die Mean Accuracy wie auf der moabb Webseite vergleichen zu können.

## change_input_windows

Hierin ist das Skript, welches mehrere Models mit verschiedenen Fenstergrößen trainiert.
- src\change_input_windows\our_parameters_changing_windows.py: Skript in welchem Models nach unseren Parametern erstellt werden, welche kleinere Input WIndows nutzen. (500 -> 100)

## data_compare

hierin sind die Skripte die genutzt wurden um die Datensätze sowohl zu vergleichen als auch genauer zu untersuchen. Mit Datensätze ist gemeint: Der BCI Competition IV 2a Datensatz in Form von a) Moabb Datenbank, b) als Download von der BCI Website in Form von unvorgeschnittenen GDF.

src\data_compare\compare.py: Daten werden verglichen, mit Plots. Zur Zeit falsche Anwendung, da der moabb Datensatz schon vor epochsiert wurde.
src\data_compare\compare_one_on_one.py: Shape der Daten zwischen lokaler GDF und moabb wird verglichen.
src\data_compare\compare_trials.py: Anzahl und Inhalt einzelner Trials eines Datensatzes wird verglichen (moabb - gdf)
src\data_compare\compare_trial_data_all_runs.py: Anzahl und Inhalt aller Runs eines Datensatzes wird verglichen (moabb - gdf)
src\data_compare\compare_trial_data_run0.pyAnzahl und Inhalt des ersten Runs eines Datensatzes wird verglichen (moabb - gdf)
src\data_compare\data_exploration_gdf.py: Erkundung des GDF formats, zum besseren Verständnins des Inhalts und der Nutzung des GDF Datensatzes.
src\data_compare\data_exploration_gdf_relabel_csv.py: Ausgabe der Daten einer GDF ins csv Format.
src\data_compare\data_exploration_gdf_relabel_plot.py: Kennzeichung in einem Plot der Relabelung einer gdf.
src\data_compare\data_exploration_moabb.py: Erkundung des moabb Datensatzes, zum besseren Verständnins des Inhalts und der Nutzung des moabb Datensatzes.
src\data_compare\gs_model_on_offline_bciIV.py: Versuch eines auf die "good subjects" trainierten Models mit dem Datensatz in gdf Form zu validieren.
src\data_compare\show_moabb.py: anzeige des Formats der Daten aus dem moabb Datensatz
src\data_compare\subject1_model_create_local.py: Model training für Subjekt 1 aus dem lokalen gdf Datensatz.

## gs_model_verification

Hierin sind die Skripte, mit denen versucht wurde das "good subjects" Model (Subjects: 1, 3, 8 & 9) zu verifizieren. Dh. dass die generalisierung mit guten Grunddaten funktioniert.

src\gs_model_verification\all_trials_subj8.py: Genauigkeitstest von Model= Test8lo.pth auf alle Trials im moabb Datensatz von Subjekt 8, zum Ziel das Training zu verifizieren.
src\gs_model_verification\all_windows.py: Genauigkeitstest von Model= Test8lo.pth auf alle Windows im moabb Datensatz von Subjekt 8, zum Ziel das Training zu verifizieren.
src\gs_model_verification\check_train_and_test_same_data.py: Test ein Model zu generieren und zu testen ob es auf den Trainingsdaten gute Testergebnisse liefert.
src\gs_model_verification\loso.py: Erneuter Test mit der "leave one out" Methode, Test der generalisierung.
src\gs_model_verification\multi_trial_prediction_test.py: Weiterer Genauigkeitstest von Model= Test8lo.pth auf alle Trials im moabb Datensatz von Subjekt 8, zum Ziel das Training zu verifizieren.
src\gs_model_verification\one_trial_input_test.py:  Test eines Models mit Groundtruth.
src\gs_model_verification\one_window_test.py: Test eines Models auf einzelne WIndows.
src\gs_model_verification\sessions.py: Evaluierung des "good subject" Models zum Versuch der Nachbildung der Trainingsaccuracy.

## load_model

Hierin wurde das laden eines Pytorch Models getestet, um Fehlerquellen die durch das laden entstehen könnten zu entdecken.

src\load_model\load_model.py: lädt und testet ein beliebiges Model.

## stream

Hierin wurde versucht einen Live Daten Stream mit einem Model auszuwerten. Der Stream, sowie die Preparation funktionieren, jedoch ist das Model und der Feed ins Model noch nicht ausgereift.

src\stream\bias_test.py: Versuch herauszufinden, ob ein geladenes Model ein voreingenommenes Bias auf eine Klasse hat.
src\stream\classify_stream.py: SKript um zu verifizieren, dass die gestreamten Daten kortrekt sind.
src\stream\gui.py: Skript in dem eine GUI entworfen wird, eine Stream empfangen wird, ein Model geladen wird und die Predictions dieses Models auf die gestreamten Daten in der GUI ausgegeben wird.
src\stream\streamer_dummy.py: Skript, um eine Testdatei zu streamen per LSL.
src\stream\stream_moabb_raw.py: Skript um die Daten aus dem moabb Datensatz als raw zu streamen.
src\stream\stream_moabb_subject.py: Test stream eines wählbaren Subjekts aus dem moabb Datensatz mit preprocessing (nicht best practice)
src\stream\stream_moab_with_marker.py: Streamt ein beliebiges Subjekt aus moabb als raw und füght einen Marker Channel hinzu.
src\stream\stream_xdf.py: streamt xdf files per LSL.
src\stream\test_moabb.py: Gibt die Klassenverteilung von Fenstern des Moabb DAtensatzes aus.
src\stream\test_xdf.py: streamt xdf files per LSL und gibt mehrere Debugs aus, zur Analyse des Datenstreams.

## torch_gpu_test.py
Hierin kann getestet werden ob pytorch eine GPU erkennt und diese nutzen könnte.