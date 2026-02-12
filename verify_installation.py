#!/usr/bin/env python3
"""
Verify Installation and Basic Functionality
Quick check that all components are properly installed
"""

import sys

print("="*70)
print("VERIFYING INSTALLATION")
print("="*70)
print()

# Check Python version
print(f"Python version: {sys.version}")
if sys.version_info < (3, 7):
    print("✗ Python 3.7+ required")
    sys.exit(1)
else:
    print("✓ Python version OK")
print()

# Check imports
print("Checking required packages...")
packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'librosa': 'Librosa',
    'sounddevice': 'SoundDevice',
    'soundfile': 'SoundFile',
    'sklearn': 'Scikit-learn',
    'joblib': 'Joblib',
}

all_ok = True
for module, name in packages.items():
    try:
        __import__(module)
        print(f"✓ {name:20s} installed")
    except ImportError:
        print(f"✗ {name:20s} NOT installed")
        all_ok = False

print()

if not all_ok:
    print("Missing packages! Install with:")
    print("  pip install -r requirements.txt")
    print()
    sys.exit(1)

# Check custom modules
print("Checking custom modules...")
try:
    from advanced_features import AdvancedFeatureExtractor
    print("✓ advanced_features.py OK")
except Exception as e:
    print(f"✗ advanced_features.py: {e}")
    all_ok = False

try:
    from retrain_model import TongueClickModelTrainer
    print("✓ retrain_model.py OK")
except Exception as e:
    print(f"✗ retrain_model.py: {e}")
    all_ok = False

try:
    from ml_detector import MLTongueClickDetector
    print("✓ ml_detector.py OK")
except Exception as e:
    print(f"✗ ml_detector.py: {e}")
    all_ok = False

print()

if not all_ok:
    print("Some modules have errors!")
    sys.exit(1)

# Test feature extraction
print("Testing feature extraction...")
try:
    import numpy as np
    extractor = AdvancedFeatureExtractor(sample_rate=44100)

    # Create 50ms of synthetic audio
    audio = np.random.randn(int(0.05 * 44100))
    features = extractor.extract_all_features(audio)
    vector = extractor.features_to_vector(features)

    print(f"✓ Feature extraction OK (extracted {len(vector)} features)")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    all_ok = False

print()

# Check audio devices
print("Checking audio devices...")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]

    if len(input_devices) > 0:
        print(f"✓ Found {len(input_devices)} input device(s)")
        print(f"  Default: {sd.query_devices(kind='input')['name']}")
    else:
        print("⚠ No input devices found (you may need to configure audio)")
except Exception as e:
    print(f"⚠ Could not check audio devices: {e}")

print()
print("="*70)

if all_ok:
    print("✓ INSTALLATION VERIFIED")
    print()
    print("Everything is ready!")
    print()
    print("Next steps:")
    print("  1. Add tongue click samples to: training_data/positives/")
    print("  2. Run: ./quick_start.sh")
    print("  or")
    print("  2. Collect negatives: python3 collect_negative_samples.py --interactive")
    print("  3. Train model: python3 retrain_model.py --positive ... --negative ...")
    print("  4. Test detector: python3 ml_detector.py --mode realtime")
    print()
else:
    print("✗ VERIFICATION FAILED")
    print("Please fix the errors above before proceeding.")

print("="*70)

sys.exit(0 if all_ok else 1)
