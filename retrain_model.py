#!/usr/bin/env python3
"""
Model Retraining Script
Retrain tongue click detector with positive and negative examples
"""

import numpy as np
import librosa
import os
import json
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from advanced_features import AdvancedFeatureExtractor
from typing import List, Tuple


class TongueClickModelTrainer:
    """Train and evaluate tongue click detection models."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.feature_extractor = AdvancedFeatureExtractor(sample_rate)
        self.scaler = StandardScaler()
        self.model = None

    def load_dataset(self, positive_dir: str, negative_dir: str,
                    chunk_duration: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process training data.

        Args:
            positive_dir: Directory with tongue click samples
            negative_dir: Directory with negative samples
            chunk_duration: Duration of each chunk in seconds

        Returns:
            X (features), y (labels)
        """
        print("\n" + "="*70)
        print("LOADING DATASET")
        print("="*70)

        X_positive = self._load_positive_samples(positive_dir, chunk_duration)
        X_negative = self._load_negative_samples(negative_dir, chunk_duration)

        print(f"\nPositive samples: {len(X_positive)}")
        print(f"Negative samples: {len(X_negative)}")

        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([
            np.ones(len(X_positive)),   # 1 = tongue click
            np.zeros(len(X_negative))   # 0 = not tongue click
        ])

        print(f"Total samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")

        return X, y

    def _load_positive_samples(self, positive_dir: str, chunk_duration: float) -> np.ndarray:
        """Load positive (tongue click) samples."""
        print(f"\nLoading positive samples from: {positive_dir}")

        features_list = []

        # Load all wav files in directory
        for root, dirs, files in os.walk(positive_dir):
            for filename in files:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filepath = os.path.join(root, filename)
                    try:
                        features = self._extract_features_from_file(filepath, chunk_duration)
                        features_list.extend(features)
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")

        print(f"  Extracted {len(features_list)} feature vectors")

        return np.array(features_list) if features_list else np.array([])

    def _load_negative_samples(self, negative_dir: str, chunk_duration: float) -> np.ndarray:
        """Load negative (non-click) samples."""
        print(f"\nLoading negative samples from: {negative_dir}")

        features_list = []

        # Load metadata if available
        metadata_file = os.path.join(negative_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            for entry in metadata:
                category = entry['category']
                filename = entry['filename']
                filepath = os.path.join(negative_dir, category, filename)

                if os.path.exists(filepath):
                    try:
                        features = self._extract_features_from_file(filepath, chunk_duration)
                        features_list.extend(features)
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")
        else:
            # No metadata, scan all wav files
            for root, dirs, files in os.walk(negative_dir):
                for filename in files:
                    if filename.endswith('.wav') or filename.endswith('.WAV'):
                        filepath = os.path.join(root, filename)
                        try:
                            features = self._extract_features_from_file(filepath, chunk_duration)
                            features_list.extend(features)
                        except Exception as e:
                            print(f"  Error loading {filename}: {e}")

        print(f"  Extracted {len(features_list)} feature vectors")

        return np.array(features_list) if features_list else np.array([])

    def _extract_features_from_file(self, filepath: str, chunk_duration: float) -> List[np.ndarray]:
        """Extract features from audio file using sliding window."""
        # Load audio
        audio, sr = librosa.load(filepath, sr=self.sample_rate)

        # Split into chunks
        chunk_samples = int(chunk_duration * self.sample_rate)
        features_list = []

        # Use 50% overlap
        hop_samples = chunk_samples // 2

        for i in range(0, len(audio) - chunk_samples, hop_samples):
            chunk = audio[i:i + chunk_samples]

            # Only process if sufficient energy
            if np.max(np.abs(chunk)) > 0.01:
                try:
                    features = self.feature_extractor.extract_all_features(chunk)
                    feature_vector = self.feature_extractor.features_to_vector(features)
                    features_list.append(feature_vector)
                except Exception as e:
                    # Skip problematic chunks
                    continue

        return features_list

    def train_model(self, X: np.ndarray, y: np.ndarray,
                   model_type: str = 'random_forest') -> None:
        """
        Train classification model.

        Args:
            X: Feature matrix
            y: Labels
            model_type: 'random_forest', 'gradient_boost', or 'svm'
        """
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Select and train model
        print(f"\nTraining {model_type} model...")

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)

        # Training accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        print(f"\nTraining accuracy: {train_score*100:.2f}%")

        # Test accuracy
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Test accuracy: {test_score*100:.2f}%")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

        # Detailed classification report
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Not Click', 'Tongue Click']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("              Not Click  Click")
        print(f"Actual Not Click   {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"       Click       {cm[1,0]:5d}   {cm[1,1]:5d}")

        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 10 Most Important Features:")
            importances = self.model.feature_importances_
            feature_names = [
                'duration_ms', 'attack_time_ms', 'decay_time_ms', 'kurtosis',
                'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                'spectral_flatness', 'spectral_flux', 'centroid_variance',
                'rms_peak', 'rms_mean', 'peak_to_mean_ratio', 'onset_strength',
                'hnr', 'has_pitch', 'zcr',
                'low_freq_ratio', 'mid_freq_ratio', 'high_freq_ratio',
                'has_ringing', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
            ]
            indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

    def save_model(self, output_dir: str = 'models'):
        """Save trained model and scaler."""
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, 'tongue_click_model.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")

    def load_model(self, model_dir: str = 'models'):
        """Load trained model and scaler."""
        model_path = os.path.join(model_dir, 'tongue_click_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain tongue click detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python retrain_model.py \\
    --positive training_data/positives \\
    --negative training_data/negatives

  # Train with SVM
  python retrain_model.py \\
    --positive training_data/positives \\
    --negative training_data/negatives \\
    --model svm

  # Custom chunk duration
  python retrain_model.py \\
    --positive training_data/positives \\
    --negative training_data/negatives \\
    --chunk-duration 0.05
        """
    )

    parser.add_argument('--positive', type=str, required=True,
                       help='Directory with positive (tongue click) samples')
    parser.add_argument('--negative', type=str, required=True,
                       help='Directory with negative samples')
    parser.add_argument('--model', choices=['random_forest', 'gradient_boost', 'svm'],
                       default='random_forest',
                       help='Model type (default: random_forest)')
    parser.add_argument('--chunk-duration', type=float, default=0.1,
                       help='Chunk duration in seconds (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for trained model')

    args = parser.parse_args()

    # Train model
    trainer = TongueClickModelTrainer()

    # Load dataset
    X, y = trainer.load_dataset(
        args.positive,
        args.negative,
        chunk_duration=args.chunk_duration
    )

    if len(X) == 0:
        print("\nERROR: No samples loaded! Check your data directories.")
        return

    # Train
    trainer.train_model(X, y, model_type=args.model)

    # Save
    trainer.save_model(args.output_dir)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
