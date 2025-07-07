import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the CSV
df = pd.read_csv("src/our_way/lsl_eegnetv4_results_3.csv")

# Filter out 'unknown' true labels
df_valid = df[df['true_label'] != 'unknown']

# Overall accuracy
acc = (df_valid['true_label'] == df_valid['pred_label']).mean()
print(f"Overall accuracy (excluding 'unknown'): {acc*100:.2f}%")

# Confusion matrix
labels = ['left_hand', 'right_hand', 'feet', 'tongue']
cm = confusion_matrix(df_valid['true_label'], df_valid['pred_label'], labels=labels)
print("Confusion matrix:")
print(pd.DataFrame(cm, index=labels, columns=labels))

# Per-class accuracy
print("\nPer-class accuracy:")
print(classification_report(df_valid['true_label'], df_valid['pred_label'], labels=labels, zero_division=0))

# Confidence summary for each class
for i, class_name in enumerate(labels):
    conf_col = f'conf_{i}'
    print(f"\nConfidence stats for {class_name}:")
    print(df_valid[conf_col].describe())