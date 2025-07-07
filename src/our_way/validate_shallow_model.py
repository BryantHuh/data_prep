# -*- coding: utf-8 -*-
"""
Validate ShallowFBCSPNet model on MOABB BNCI2014_001 subject 3 test session.
This script loads the trained model and evaluates it on the test dataset,
providing detailed performance metrics and visualizations.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.models import ShallowFBCSPNet
from braindecode.visualization import plot_confusion_matrix
from braindecode.preprocessing import exponential_moving_standardize

def load_and_preprocess_data(subject_id=3):
    """Load and preprocess the test dataset."""
    print(f"Loading MOABB dataset for subject {subject_id}...")

    # Load dataset
    dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])

    # Channel selection (16 OpenBCI channels)
    included_channels = [
        'C3', 'C4', 'Cz',
        'FC1', 'FC2', 'FCz',
        'CP1', 'CP2', 'CPz',
        'P1', 'P2', 'Pz',
        'C1', 'C2',
        'CP3', 'CP4'
    ]

    # Preprocessing pipeline (exactly as in training)
    preprocessors = [
        Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
        Preprocessor(lambda data: data * 1e6),  # Scale to microvolts
        Preprocessor('resample', sfreq=125),
        Preprocessor('filter', l_freq=4, h_freq=38),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=250
        )
    ]

    print("Applying preprocessing...")
    preprocess(dataset, preprocessors, n_jobs=1)

    # Create windows
    input_window_samples = 250
    sfreq = dataset.datasets[0].raw.info['sfreq']
    trial_start_offset_samples = int(-0.5 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=1,  # For validation, we want all windows
        drop_last_window=False,
        preload=True
    )

    # Split to get test set
    splitted = windows_dataset.split('session')
    if '1test' not in splitted:
        raise ValueError(f"No test session found for subject {subject_id}")

    test_set = splitted['1test']

    print(f"Test set created with {len(test_set)} windows")
    print(f"Window shape: {test_set[0][0].shape}")

    return test_set

def load_model(model_path, device):
    """Load the trained ShallowFBCSPNet model."""
    print(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load model
    torch.serialization.add_safe_globals([ShallowFBCSPNet])
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()

    print(f"Model loaded successfully on {device}")
    return model

def predict_on_dataset(model, test_set, device):
    """Run predictions on the test dataset."""
    print("Running predictions...")

    predictions = []
    confidences = []
    probabilities_list = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for i, (x, y, *rest) in enumerate(test_set):
            # Convert to tensor
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

            # Get prediction
            logits = model(x_tensor)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)  # Average over time dimension for cropped decoding

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probs, 1)

            # Store results
            predictions.append(predicted_class.cpu().numpy()[0])
            confidences.append(confidence.cpu().numpy()[0])
            probabilities_list.append(probs.cpu().numpy()[0])
            true_labels.append(y)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_set)} windows")

    return np.array(predictions), np.array(confidences), np.array(probabilities_list), np.array(true_labels)

def analyze_results(predictions, confidences, probabilities, true_labels, test_set):
    """Analyze and display results."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    # Get label mapping
    if hasattr(test_set.datasets[0], 'window_kwargs') and test_set.datasets[0].window_kwargs:
        label_dict = test_set.datasets[0].window_kwargs[0][1]['mapping']
        labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
    else:
        labels = ['feet', 'left_hand', 'right_hand', 'tongue']

    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print(classification_report(true_labels, predictions, target_names=labels, digits=3))

    # Confidence analysis
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Std confidence: {np.std(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")
    print(f"  Max confidence: {np.max(confidences):.3f}")

    # Confidence by class
    print(f"\nConfidence by Class:")
    for i, label in enumerate(labels):
        class_mask = true_labels == i
        if np.any(class_mask):
            class_conf = confidences[class_mask]
            print(f"  {label}: {np.mean(class_conf):.3f} ± {np.std(class_conf):.3f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    return accuracy, cm, labels, confidences

def create_visualizations(accuracy, cm, labels, confidences, probabilities, save_dir):
    """Create and save visualizations."""
    print(f"\nCreating visualizations...")

    # 1. Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], 'd'),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")

    ax_cm.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xticklabels=labels, yticklabels=labels,
              title=f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)',
              ylabel='True label',
              xlabel='Predicted label')

    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'validation_confusion_matrix.png')
    fig_cm.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    print(f"  Confusion matrix saved to {cm_path}")

    # 2. Confidence Distribution
    fig_conf, ax_conf = plt.subplots(figsize=(10, 6))

    # Histogram of all confidences
    ax_conf.hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax_conf.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
    ax_conf.set_xlabel('Confidence')
    ax_conf.set_ylabel('Frequency')
    ax_conf.set_title('Distribution of Prediction Confidence')
    ax_conf.legend()
    ax_conf.grid(True, alpha=0.3)

    plt.tight_layout()
    conf_path = os.path.join(save_dir, 'validation_confidence_distribution.png')
    fig_conf.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close(fig_conf)
    print(f"  Confidence distribution saved to {conf_path}")

    # 3. Class Probability Heatmap
    fig_prob, ax_prob = plt.subplots(figsize=(12, 8))

    # Calculate mean probabilities for each class
    mean_probs = np.mean(probabilities, axis=0)
    prob_matrix = probabilities.T  # Shape: (n_classes, n_samples)

    im = ax_prob.imshow(prob_matrix, aspect='auto', cmap='viridis')
    ax_prob.set_yticks(range(len(labels)))
    ax_prob.set_yticklabels(labels)
    ax_prob.set_xlabel('Sample Index')
    ax_prob.set_ylabel('Class')
    ax_prob.set_title('Class Probabilities Over All Samples')

    # Add colorbar
    cbar = ax_prob.figure.colorbar(im, ax=ax_prob)
    cbar.set_label('Probability')

    plt.tight_layout()
    prob_path = os.path.join(save_dir, 'validation_class_probabilities.png')
    fig_prob.savefig(prob_path, dpi=300, bbox_inches='tight')
    plt.close(fig_prob)
    print(f"  Class probabilities saved to {prob_path}")

def save_results(predictions, confidences, probabilities, true_labels, labels, save_dir):
    """Save detailed results to CSV."""
    print(f"\nSaving detailed results...")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'true_label': [labels[y] for y in true_labels],
        'predicted_label': [labels[p] for p in predictions],
        'confidence': confidences,
        'correct': predictions == true_labels
    })

    # Add individual class probabilities
    for i, label in enumerate(labels):
        results_df[f'prob_{label}'] = probabilities[:, i]

    # Save to CSV
    results_path = os.path.join(save_dir, 'validation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  Detailed results saved to {results_path}")

    # Save summary statistics
    summary = {
        'total_samples': len(predictions),
        'accuracy': accuracy_score(true_labels, predictions),
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences)
    }

    summary_path = os.path.join(save_dir, 'validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("VALIDATION SUMMARY\n")
        f.write("="*50 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"  Summary saved to {summary_path}")

def main():
    """Main validation function."""
    print("ShallowFBCSPNet Model Validation")
    print("="*50)

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')
    save_dir = os.path.join(project_root, 'log', 'validation')
    os.makedirs(save_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load and preprocess data
        test_set = load_and_preprocess_data(subject_id=3)

        # Load model
        model = load_model(model_path, device)

        # Run predictions
        predictions, confidences, probabilities, true_labels = predict_on_dataset(
            model, test_set, device
        )

        # Analyze results
        accuracy, cm, labels, confidences = analyze_results(
            predictions, confidences, probabilities, true_labels, test_set
        )

        # Create visualizations
        create_visualizations(
            accuracy, cm, labels, confidences, probabilities, save_dir
        )

        # Save results
        save_results(predictions, confidences, probabilities, true_labels, labels, save_dir)

        print(f"\n✅ Validation completed successfully!")
        print(f"Results saved to: {save_dir}")
        print(f"Final accuracy: {accuracy*100:.2f}%")

    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()