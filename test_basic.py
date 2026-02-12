#!/usr/bin/env python3
"""
Basic test script to verify the tongue click detector can be imported
and initialized without errors.
"""

import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        import scipy
        print("✓ scipy imported")
    except ImportError as e:
        print(f"✗ scipy import failed: {e}")
        return False

    try:
        import sounddevice as sd
        print("✓ sounddevice imported")
    except ImportError as e:
        print(f"✗ sounddevice import failed: {e}")
        return False

    try:
        import librosa
        print("✓ librosa imported")
    except ImportError as e:
        print(f"✗ librosa import failed: {e}")
        return False

    return True


def test_detector_init():
    """Test that the TongueClickDetector can be initialized."""
    print("\nTesting detector initialization...")

    try:
        from tongue_click_detector import DetectorFactory, DetectorConfig
        print("✓ DetectorFactory imported")

        detector = DetectorFactory.create_default_detector()
        print("✓ TongueClickDetector initialized with defaults")

        config = DetectorConfig(sample_rate=16000, threshold=6.0)
        detector2 = DetectorFactory.create_default_detector(config)
        print("✓ TongueClickDetector initialized with custom parameters")

        return True
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        return False


def test_audio_devices():
    """Test that audio devices can be listed."""
    print("\nTesting audio device detection...")

    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✓ Found {len(devices)} audio devices")
        return True
    except Exception as e:
        print(f"✗ Audio device detection failed: {e}")
        return False


def test_synthetic_audio():
    """Test detection on synthetic audio data."""
    print("\nTesting with synthetic audio...")

    try:
        import numpy as np
        from tongue_click_detector import DetectorFactory

        detector = DetectorFactory.create_default_detector()

        # Generate a simple synthetic audio chunk (440 Hz sine wave)
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(detector.config.sample_rate * duration))
        audio_chunk = np.sin(2 * np.pi * 440 * t)

        # Try to detect (should not detect a sine wave as a click)
        result = detector.detect(audio_chunk)
        print(f"✓ Synthetic audio processed (click={result.is_click}, score={result.confidence_score:.3f})")

        return True
    except Exception as e:
        print(f"✗ Synthetic audio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("TONGUE CLICK DETECTOR - BASIC TESTS")
    print("=" * 70 + "\n")

    all_passed = True

    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed. Please install dependencies:")
        print("  pip install -r requirements.txt")
        all_passed = False

    # Test detector initialization
    if not test_detector_init():
        all_passed = False

    # Test audio devices
    if not test_audio_devices():
        print("  Note: This might be expected in some environments")

    # Test with synthetic audio
    if not test_synthetic_audio():
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("The tongue click detector is ready to use!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above and install missing dependencies.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
