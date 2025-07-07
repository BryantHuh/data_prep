# -*- coding: utf-8 -*-
"""
Simple test to verify validation script works
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import validate_shallow_model
    print("✅ Validation script imports successfully")

    # Check if model exists
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

    if os.path.exists(model_path):
        print(f"✅ Model found at: {model_path}")
    else:
        print(f"❌ Model not found at: {model_path}")
        print("Available models:")
        model_dir = os.path.join(project_root, 'models')
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.pth'):
                    print(f"  - {file}")

    print("Ready to run validation!")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")