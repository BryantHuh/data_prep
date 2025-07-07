# -*- coding: utf-8 -*-
"""
Console runner for the real-time BCI classifier.
Simple script to run the classifier with console output.
"""

from realtime_classifier import RealtimeBCIClassifier

def main():
    """Run the real-time classifier with console output."""
    print("=== Real-time BCI Classifier ===")
    print("Starting console mode...")

    # Initialize classifier
    classifier = RealtimeBCIClassifier(subject_id=3)

    # Run with console output
    classifier.run_console()

    # Print final statistics
    accuracy = classifier.get_accuracy()
    print(f"\nFinal accuracy: {accuracy*100:.2f}%")

    # Print confusion matrix
    cm_result = classifier.get_confusion_matrix()
    if cm_result:
        cm, labels = cm_result
        print("\nConfusion Matrix:")
        print("True\\Pred", end="")
        for label in labels:
            print(f"{label:>10}", end="")
        print()
        for i, true_label in enumerate(labels):
            print(f"{true_label:>8}", end="")
            for j in range(len(labels)):
                print(f"{cm[i,j]:>10}", end="")
            print()

if __name__ == "__main__":
    main()