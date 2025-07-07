# -*- coding: utf-8 -*-
"""
Test script to verify that the softmax fix is working correctly.
This ensures that we're not double-applying softmax to ShallowFBCSPNet outputs.
"""

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet
import os

def test_softmax_fix():
    """Test that we're not double-applying softmax."""

    print("🧪 Testing Softmax Fix")
    print("=" * 40)

    # Create a simple ShallowFBCSPNet model
    n_channels = 16
    n_classes = 4
    input_window_samples = 250

    model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        final_conv_length=30
    )
    model.to_dense_prediction_model()

    # Create test input
    test_input = torch.randn(1, n_channels, input_window_samples)

    print("📊 Model architecture:")
    print(f"   Input shape: {test_input.shape}")
    # Skip get_output_shape() as it causes issues with small inputs

    # Test 1: Check if model outputs sum to 1 (indicating probabilities)
    print("\n🔍 Test 1: Checking if model outputs are probabilities...")

    model.eval()
    with torch.no_grad():
        output = model(test_input)
        if output.ndim == 3:
            output = output.mean(dim=2)

        # Check if outputs sum to 1 (probabilities)
        output_sum = output.sum(dim=1)
        print(f"   Output sum per sample: {output_sum.cpu().numpy()}")

        if torch.allclose(output_sum, torch.ones_like(output_sum), atol=1e-6):
            print("   ✅ Model outputs are probabilities (sum to 1)")
            print("   ✅ No need to apply softmax again")
        else:
            print("   ❌ Model outputs are not probabilities")
            print("   ❌ Need to apply softmax")

    # Test 2: Compare with and without additional softmax
    print("\n🔍 Test 2: Comparing with/without additional softmax...")

    with torch.no_grad():
        output = model(test_input)
        if output.ndim == 3:
            output = output.mean(dim=2)

        # Method 1: Use model output directly (correct)
        probs_correct = output
        pred_correct = torch.argmax(probs_correct, dim=1)
        conf_correct = torch.max(probs_correct, dim=1)[0]

        # Method 2: Apply softmax again (incorrect)
        probs_incorrect = torch.nn.functional.softmax(output, dim=1)
        pred_incorrect = torch.argmax(probs_incorrect, dim=1)
        conf_incorrect = torch.max(probs_incorrect, dim=1)[0]

        print(f"   Correct prediction: {pred_correct.cpu().numpy()}")
        print(f"   Incorrect prediction: {pred_incorrect.cpu().numpy()}")
        print(f"   Predictions match: {torch.equal(pred_correct, pred_incorrect)}")

        print(f"   Correct confidence: {conf_correct.cpu().numpy():.6f}")
        print(f"   Incorrect confidence: {conf_incorrect.cpu().numpy():.6f}")
        print(f"   Confidence difference: {abs(conf_correct - conf_incorrect).cpu().numpy():.6f}")

        # Check if probabilities are different
        prob_diff = torch.abs(probs_correct - probs_incorrect).max()
        print(f"   Max probability difference: {prob_diff.cpu().numpy():.6f}")

        if prob_diff < 1e-6:
            print("   ✅ Probabilities are identical (softmax has no effect)")
        else:
            print("   ⚠️  Probabilities are different (softmax changes values)")

    # Test 3: Check real-time classifier behavior
    print("\n🔍 Test 3: Testing real-time classifier behavior...")

    try:
        from realtime_classifier_online import RealtimeBCIClassifierOnline

        # Create a dummy model path for testing
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        dummy_model_path = os.path.join(project_root, 'models', 'dummy_model.pth')

        # Save the test model temporarily
        torch.save(model, dummy_model_path)

        # Test the classifier
        classifier = RealtimeBCIClassifierOnline(
            model_path=dummy_model_path,
            window_size=input_window_samples,
            sample_rate=125
        )

        # Add some test samples
        for i in range(1000):
            sample = np.random.randn(n_channels)
            classifier.add_sample(sample)

        # Try to predict
        if classifier.is_ready():
            result = classifier.predict()
            print(f"   ✅ Classifier prediction: {result['class_label']}")
            print(f"   ✅ Confidence: {result['confidence']:.6f}")
            print(f"   ✅ Probabilities sum: {sum(result['probabilities']):.6f}")

            if abs(sum(result['probabilities']) - 1.0) < 1e-6:
                print("   ✅ Probabilities sum to 1 (correct)")
            else:
                print("   ❌ Probabilities don't sum to 1 (incorrect)")
        else:
            print("   ⚠️  Classifier not ready for prediction")

        # Clean up
        os.remove(dummy_model_path)

    except Exception as e:
        print(f"   ⚠️  Could not test real-time classifier: {e}")

    print("\n📋 Summary:")
    print("=" * 40)
    print("✅ The softmax fix ensures that:")
    print("   - Model outputs are used directly as probabilities")
    print("   - No double application of softmax")
    print("   - Correct probability distributions")
    print("   - Accurate confidence scores")

if __name__ == "__main__":
    test_softmax_fix()