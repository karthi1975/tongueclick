#!/usr/bin/env python3
"""
Test Suite for ML Training Pipeline
Tests feature extraction, data loading, and model training
"""

import unittest
import numpy as np
import tempfile
import os
import json
import soundfile as sf
from advanced_features import AdvancedFeatureExtractor, AdvancedAudioFeatures
from retrain_model import TongueClickModelTrainer


class TestFeatureExtraction(unittest.TestCase):
    """Test advanced feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 44100
        self.extractor = AdvancedFeatureExtractor(self.sample_rate)

    def test_normalize(self):
        """Test audio normalization."""
        audio = np.array([0.5, -0.3, 0.8, -0.2])
        normalized = self.extractor._normalize(audio)

        # Should be normalized to [-1, 1]
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0)
        self.assertTrue(np.all(np.abs(normalized) <= 1.0))

    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        # Create synthetic audio (100ms)
        duration = 0.1
        audio = np.random.randn(int(self.sample_rate * duration))

        features = self.extractor._extract_temporal_features(audio)

        # Check required keys
        self.assertIn('duration_ms', features)
        self.assertIn('attack_time_ms', features)
        self.assertIn('decay_time_ms', features)
        self.assertIn('kurtosis', features)

        # Check duration is approximately correct
        self.assertAlmostEqual(features['duration_ms'], 100, delta=5)

    def test_extract_spectral_features(self):
        """Test spectral feature extraction."""
        # Create synthetic audio
        duration = 0.1
        audio = np.random.randn(int(self.sample_rate * duration))

        features = self.extractor._extract_spectral_features(audio)

        # Check required keys
        self.assertIn('centroid', features)
        self.assertIn('bandwidth', features)
        self.assertIn('rolloff', features)
        self.assertIn('flatness', features)
        self.assertIn('flux', features)

        # Check values are in reasonable range
        self.assertGreater(features['centroid'], 0)
        self.assertLess(features['centroid'], self.sample_rate / 2)
        self.assertGreater(features['flatness'], 0)
        self.assertLess(features['flatness'], 1)

    def test_extract_frequency_distribution(self):
        """Test frequency distribution analysis."""
        # Create audio with mostly high-frequency content
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # 4kHz sine wave (high frequency)
        audio = np.sin(2 * np.pi * 4000 * t)

        features = self.extractor._extract_frequency_distribution(audio)

        # Should have more high-frequency energy
        self.assertGreater(features['high'], features['low'])

    def test_extract_all_features(self):
        """Test extraction of all features."""
        # Create synthetic audio
        duration = 0.05  # 50ms (typical tongue click)
        audio = np.random.randn(int(self.sample_rate * duration))

        features = self.extractor.extract_all_features(audio)

        # Check it returns AdvancedAudioFeatures object
        self.assertIsInstance(features, AdvancedAudioFeatures)

        # Check all attributes exist
        self.assertIsNotNone(features.duration_ms)
        self.assertIsNotNone(features.spectral_flatness)
        self.assertIsNotNone(features.kurtosis)
        self.assertIsNotNone(features.low_freq_ratio)

    def test_features_to_vector(self):
        """Test conversion of features to numpy array."""
        # Create synthetic audio
        duration = 0.05
        audio = np.random.randn(int(self.sample_rate * duration))

        features = self.extractor.extract_all_features(audio)
        vector = self.extractor.features_to_vector(features)

        # Check it's a numpy array
        self.assertIsInstance(vector, np.ndarray)

        # Check it has expected length (26 features)
        self.assertEqual(len(vector), 26)

        # Check all values are numeric
        self.assertTrue(np.all(np.isfinite(vector)))


class TestDataLoading(unittest.TestCase):
    """Test dataset loading and processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 44100
        self.trainer = TongueClickModelTrainer(self.sample_rate)

        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.positive_dir = os.path.join(self.temp_dir, 'positives')
        self.negative_dir = os.path.join(self.temp_dir, 'negatives')

        os.makedirs(self.positive_dir)
        os.makedirs(self.negative_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_synthetic_audio_file(self, filepath, duration=0.5):
        """Create a synthetic audio file."""
        audio = np.random.randn(int(self.sample_rate * duration))
        sf.write(filepath, audio, self.sample_rate)

    def test_load_positive_samples(self):
        """Test loading positive samples."""
        # Create 3 synthetic positive samples
        for i in range(3):
            filepath = os.path.join(self.positive_dir, f'click_{i}.wav')
            self.create_synthetic_audio_file(filepath, duration=0.2)

        # Load samples
        features = self.trainer._load_positive_samples(self.positive_dir, chunk_duration=0.1)

        # Should have extracted multiple feature vectors
        self.assertGreater(len(features), 0)
        self.assertEqual(features.shape[1], 26)  # 26 features

    def test_load_negative_samples_with_metadata(self):
        """Test loading negative samples with metadata."""
        # Create category directory
        category_dir = os.path.join(self.negative_dir, 'test_category')
        os.makedirs(category_dir)

        # Create synthetic file
        filename = 'test_sound.wav'
        filepath = os.path.join(category_dir, filename)
        self.create_synthetic_audio_file(filepath, duration=0.2)

        # Create metadata
        metadata = [{
            'filename': filename,
            'category': 'test_category',
            'label': 0
        }]

        metadata_path = os.path.join(self.negative_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Load samples
        features = self.trainer._load_negative_samples(self.negative_dir, chunk_duration=0.1)

        # Should have extracted features
        self.assertGreater(len(features), 0)

    def test_load_dataset_balanced(self):
        """Test loading balanced dataset."""
        # Create 2 positive samples
        for i in range(2):
            filepath = os.path.join(self.positive_dir, f'click_{i}.wav')
            self.create_synthetic_audio_file(filepath, duration=0.2)

        # Create 2 negative samples
        category_dir = os.path.join(self.negative_dir, 'test')
        os.makedirs(category_dir)
        for i in range(2):
            filepath = os.path.join(category_dir, f'negative_{i}.wav')
            self.create_synthetic_audio_file(filepath, duration=0.2)

        # Load dataset
        X, y = self.trainer.load_dataset(self.positive_dir, self.negative_dir,
                                         chunk_duration=0.1)

        # Check we got features and labels
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))

        # Check we have both classes
        self.assertTrue(np.any(y == 1))  # Positive class
        self.assertTrue(np.any(y == 0))  # Negative class


class TestModelTraining(unittest.TestCase):
    """Test model training (lightweight test with small synthetic dataset)."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 44100
        self.trainer = TongueClickModelTrainer(self.sample_rate)

    def test_train_random_forest(self):
        """Test Random Forest training."""
        # Create small synthetic dataset
        n_samples = 50
        n_features = 26

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)  # Binary labels

        # Train model
        self.trainer.train_model(X, y, model_type='random_forest')

        # Check model exists
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.scaler)

        # Check model can make predictions
        X_scaled = self.trainer.scaler.transform(X[:5])
        predictions = self.trainer.model.predict(X_scaled)

        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Create and train simple model
        X = np.random.randn(50, 26)
        y = np.random.randint(0, 2, 50)
        self.trainer.train_model(X, y, model_type='random_forest')

        # Save model
        temp_dir = tempfile.mkdtemp()
        self.trainer.save_model(temp_dir)

        # Check files exist
        model_path = os.path.join(temp_dir, 'tongue_click_model.pkl')
        scaler_path = os.path.join(temp_dir, 'scaler.pkl')

        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(scaler_path))

        # Load model in new trainer
        new_trainer = TongueClickModelTrainer(self.sample_rate)
        new_trainer.load_model(temp_dir)

        self.assertIsNotNone(new_trainer.model)
        self.assertIsNotNone(new_trainer.scaler)

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def run_tests():
    """Run all tests."""
    print("="*70)
    print("RUNNING ML PIPELINE TESTS")
    print("="*70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
