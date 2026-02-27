#!/usr/bin/env python3
"""
ML-Based Tongue Click Detector
Uses trained machine learning model for robust detection
"""

import numpy as np
import sounddevice as sd
import joblib
import argparse
import time
from advanced_features import AdvancedFeatureExtractor
from typing import Optional, Callable, List


class MLTongueClickDetector:
    """Tongue click detector using trained ML model. Memory-safe for 24h+ sessions."""

    # Keep only the most recent N detections in memory
    MAX_DETECTIONS_IN_MEMORY = 10000

    def __init__(self, model_path: str = 'models/tongue_click_model.pkl',
                 scaler_path: str = 'models/scaler.pkl',
                 sample_rate: int = 44100,
                 confidence_threshold: float = 0.7,
                 min_energy: float = 0.02):
        """
        Initialize ML-based detector.

        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            sample_rate: Audio sample rate
            confidence_threshold: Minimum confidence for detection (0-1)
            min_energy: Minimum energy threshold to process audio
        """
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.min_energy = min_energy

        # Load model and scaler
        print("Loading ML model...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("✓ Model loaded successfully")

        # Initialize feature extractor
        self.feature_extractor = AdvancedFeatureExtractor(sample_rate)

        # Rate limiting (prevent too frequent detections)
        self.last_detection_time = 0
        self.min_detection_interval = 0.15  # 150ms minimum between clicks

    def detect(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        Detect tongue click in audio chunk.

        Args:
            audio_chunk: Audio data

        Returns:
            (is_click, confidence)
        """
        # Check energy threshold
        if np.max(np.abs(audio_chunk)) < self.min_energy:
            return False, 0.0

        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(audio_chunk)
            feature_vector = self.feature_extractor.features_to_vector(features)

            # Scale features
            feature_scaled = self.scaler.transform([feature_vector])

            # Predict
            if hasattr(self.model, 'predict_proba'):
                # Get probability for click class (class 1)
                probabilities = self.model.predict_proba(feature_scaled)[0]
                confidence = float(probabilities[1])  # Probability of being a click
            else:
                # Model doesn't support probabilities, use decision function
                prediction = self.model.predict(feature_scaled)[0]
                confidence = 1.0 if prediction == 1 else 0.0

            is_click = confidence >= self.confidence_threshold

            return is_click, confidence

        except Exception as e:
            print(f"Error during detection: {e}")
            return False, 0.0

    def real_time_detection(self, duration: int = 10,
                           callback: Optional[Callable] = None,
                           device: Optional[int] = None) -> List[dict]:
        """
        Perform real-time tongue click detection.
        Memory-safe for long-running sessions (24h+).

        Args:
            duration: Duration to listen in seconds
            callback: Optional callback function(timestamp, confidence)

        Returns:
            List of most recent detected clicks with timestamps and confidence
        """
        import gc
        import threading
        import queue

        print(f"\n{'='*70}")
        print("ML-BASED TONGUE CLICK DETECTION")
        print(f"{'='*70}")
        print(f"\nListening for {duration} seconds...")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("\nMake tongue click sounds!")
        print("Press Ctrl+C to stop early\n", flush=True)

        detections = []
        total_click_count = [0]
        gc_counter = [0]
        start_time = time.time()
        audio_queue = queue.Queue(maxsize=50)
        stop_event = threading.Event()

        # Ensure chunk has enough samples for FFT (n_fft=2048)
        # At 16000Hz, need at least 2048 samples = 128ms
        min_chunk_samples = 2048
        chunk_duration = max(0.1, min_chunk_samples / self.sample_rate + 0.01)
        chunk_samples = int(self.sample_rate * chunk_duration)

        def audio_callback(indata, frames, time_info, status):
            """Quickly copy audio to queue — no heavy processing here."""
            try:
                audio_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # Drop chunk if processing can't keep up

        def processing_thread():
            """Process audio chunks from queue in a separate thread."""
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                is_click, confidence = self.detect(audio_chunk)

                if is_click:
                    current_time = time.time()
                    timestamp = current_time - start_time

                    if current_time - self.last_detection_time >= self.min_detection_interval:
                        self.last_detection_time = current_time

                        detection = {
                            'timestamp': timestamp,
                            'confidence': confidence
                        }
                        detections.append(detection)
                        total_click_count[0] += 1

                        print(f"✓ CLICK detected at {timestamp:.2f}s "
                              f"(confidence: {confidence:.2%})", flush=True)

                        if callback:
                            callback(timestamp, confidence)

                        if len(detections) > self.MAX_DETECTIONS_IN_MEMORY:
                            del detections[:self.MAX_DETECTIONS_IN_MEMORY // 2]

                gc_counter[0] += 1
                if gc_counter[0] >= 600:
                    gc_counter[0] = 0
                    gc.collect()

        processor = threading.Thread(target=processing_thread, daemon=True)
        processor.start()

        try:
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=chunk_samples,
                              device=device):
                sd.sleep(int(duration * 1000))
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            stop_event.set()
            processor.join(timeout=2)

        print(f"\n{'='*70}")
        print(f"Total clicks detected: {total_click_count[0]}")
        print(f"{'='*70}\n")

        return detections

    def analyze_file(self, filepath: str,
                    callback: Optional[Callable] = None) -> List[dict]:
        """
        Analyze audio file for tongue clicks.

        Args:
            filepath: Path to audio file
            callback: Optional callback function(timestamp, confidence)

        Returns:
            List of detected clicks
        """
        import librosa

        print(f"\n{'='*70}")
        print("FILE ANALYSIS")
        print(f"{'='*70}")
        print(f"\nAnalyzing: {filepath}")

        # Load audio
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        print(f"Confidence threshold: {self.confidence_threshold}\n")

        detections = []
        min_chunk_samples = 2048
        chunk_duration = max(0.1, min_chunk_samples / self.sample_rate + 0.01)
        chunk_samples = int(chunk_duration * self.sample_rate)
        hop_samples = chunk_samples // 2  # 50% overlap

        for i in range(0, len(audio) - chunk_samples, hop_samples):
            chunk = audio[i:i + chunk_samples]
            is_click, confidence = self.detect(chunk)

            if is_click:
                timestamp = i / self.sample_rate

                # Avoid duplicate detections (from overlap)
                if not detections or timestamp - detections[-1]['timestamp'] > 0.1:
                    detection = {
                        'timestamp': timestamp,
                        'confidence': confidence
                    }
                    detections.append(detection)

                    print(f"Click at {timestamp:.2f}s (confidence: {confidence:.2%})")

                    if callback:
                        callback(timestamp, confidence)

        print(f"\n{'='*70}")
        print(f"Total clicks detected: {len(detections)}")
        print(f"{'='*70}\n")

        return detections


def main():
    parser = argparse.ArgumentParser(
        description="ML-based tongue click detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time detection (default 10 seconds)
  python ml_detector.py --mode realtime

  # Real-time with custom duration and threshold
  python ml_detector.py --mode realtime --duration 30 --threshold 0.8

  # Analyze audio file
  python ml_detector.py --mode file --input recording.wav

  # Use custom model
  python ml_detector.py --mode realtime --model-dir custom_models
        """
    )

    parser.add_argument('--mode', choices=['realtime', 'file'],
                       default='realtime',
                       help='Detection mode')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration for real-time detection (seconds)')
    parser.add_argument('--input', type=str,
                       help='Input audio file (for file mode)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold (0-1, default: 0.7)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing model files')
    parser.add_argument('--min-energy', type=float, default=0.02,
                       help='Minimum energy threshold')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Audio sample rate in Hz (default: 44100, Jabra uses 16000)')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio input device index (run with --list-devices to see options)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')

    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        return

    # Initialize detector
    try:
        detector = MLTongueClickDetector(
            model_path=f"{args.model_dir}/tongue_click_model.pkl",
            scaler_path=f"{args.model_dir}/scaler.pkl",
            sample_rate=args.sample_rate,
            confidence_threshold=args.threshold,
            min_energy=args.min_energy
        )
    except Exception as e:
        print(f"\nERROR: Could not load model: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python retrain_model.py --positive <pos_dir> --negative <neg_dir>")
        return

    # Run detection
    if args.mode == 'realtime':
        detections = detector.real_time_detection(duration=args.duration, device=args.device)

    elif args.mode == 'file':
        if not args.input:
            print("ERROR: --input required for file mode")
            return

        detections = detector.analyze_file(args.input)

    # Summary
    if detections:
        avg_confidence = np.mean([d['confidence'] for d in detections])
        print(f"\nAverage confidence: {avg_confidence:.2%}")

        if len(detections) > 0:
            timestamps = [f"{d['timestamp']:.2f}s" for d in detections[:5]]
            print(f"Timestamps: {timestamps}")
            if len(detections) > 5:
                print(f"... and {len(detections) - 5} more")


if __name__ == "__main__":
    main()
