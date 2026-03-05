#!/usr/bin/env python3
"""
Call For Attention - Tongue Click Webhook Trigger

Continuously listens for tongue clicks using the trained scikit-learn
Random Forest model. After detecting consecutive clicks, triggers a
Home Assistant webhook to fire an alarm/bell.

Usage:
    # Run with defaults (3 consecutive clicks, 0.7 confidence)
    python call_for_attention.py

    # Stricter confidence
    python call_for_attention.py --threshold 0.85

    # Require 5 consecutive clicks
    python call_for_attention.py --consecutive 5

    # Use Jabra mic at 16kHz with 16k model
    python call_for_attention.py --device 2 --sample-rate 16000 --model-dir models_16k

    # Custom webhook URL
    python call_for_attention.py --webhook-url https://your-ha-instance/api/webhook/your_hook

    # List audio devices
    python call_for_attention.py --list-devices
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
import time
import signal
import sys
import gc
import threading
import queue
import requests
from datetime import datetime
from pathlib import Path
from advanced_features import AdvancedFeatureExtractor
import joblib


class CallForAttention:
    """
    Continuous tongue click listener that triggers Home Assistant webhook
    after detecting consecutive clicks.
    """

    def __init__(self, model_path='models/tongue_click_model.pkl',
                 scaler_path='models/scaler.pkl',
                 sample_rate=44100,
                 confidence_threshold=0.7,
                 min_energy=0.02,
                 webhook_url='https://ut-beachhome.homeadapt.us/api/webhook/tongue_click_alert',
                 consecutive_required=3,
                 reset_timeout=2.0,
                 debounce_interval=0.5,
                 save_clicks=True,
                 device=None):
        """
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            sample_rate: Audio sample rate (44100 for standard mic, 16000 for Jabra)
            confidence_threshold: Minimum confidence for detection (0-1)
            min_energy: Minimum energy threshold to process audio
            webhook_url: Home Assistant webhook URL
            consecutive_required: Number of consecutive clicks to trigger webhook
            reset_timeout: Seconds without click before counter resets
            debounce_interval: Minimum seconds between click detections
            save_clicks: Whether to save detected click audio files
            device: Audio input device index
        """
        self.sample_rate = sample_rate
        self.model_sample_rate = 44100
        self.confidence_threshold = confidence_threshold
        self.min_energy = min_energy
        self.webhook_url = webhook_url
        self.consecutive_required = consecutive_required
        self.reset_timeout = reset_timeout
        self.debounce_interval = debounce_interval
        self.save_clicks = save_clicks
        self.device = device

        # Load model and scaler
        print("Loading ML model...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"  Model loaded from: {model_path}")

        # Feature extractor at model's training sample rate
        self.feature_extractor = AdvancedFeatureExtractor(self.model_sample_rate)

        # State
        self.consecutive_count = 0
        self.total_clicks = 0
        self.total_triggers = 0
        self.filtered_count = 0
        self.last_click_time = 0
        self.last_detection_time = 0
        self.start_time = None
        self.restart_count = 0
        self.running = True

        # Audio processing queue (threaded to prevent input overflow)
        self.audio_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()

        # Chunk sizing (ensure enough samples for FFT n_fft=2048)
        min_chunk_samples = 2048
        self.chunk_duration = max(0.15, min_chunk_samples / self.sample_rate + 0.01)
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Save directory for detected clicks
        if self.save_clicks:
            self.save_dir = Path("detected_clicks")
            self.save_dir.mkdir(exist_ok=True)

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\nShutdown signal received...")
        self.running = False
        self.stop_event.set()

    def detect(self, audio_chunk):
        """
        Detect tongue click in audio chunk.

        Returns:
            (is_click, confidence)
        """
        if np.max(np.abs(audio_chunk)) < self.min_energy:
            return False, 0.0

        try:
            # Resample to model's training rate if needed
            if self.sample_rate != self.model_sample_rate:
                import librosa
                audio_chunk = librosa.resample(
                    audio_chunk, orig_sr=self.sample_rate,
                    target_sr=self.model_sample_rate
                )

            # Extract features
            features = self.feature_extractor.extract_all_features(audio_chunk)
            feature_vector = self.feature_extractor.features_to_vector(features)

            # Scale and predict
            feature_scaled = self.scaler.transform([feature_vector])

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_scaled)[0]
                confidence = float(probabilities[1])
            else:
                prediction = self.model.predict(feature_scaled)[0]
                confidence = 1.0 if prediction == 1 else 0.0

            is_click = confidence >= self.confidence_threshold
            return is_click, confidence

        except Exception:
            return False, 0.0

    def _reset_counter(self):
        """Reset the consecutive counter."""
        if self.consecutive_count > 0:
            print(f"  TIMEOUT - Reset counter "
                  f"(was at {self.consecutive_count}/{self.consecutive_required})",
                  flush=True)
            self.consecutive_count = 0

    def _check_timeout(self):
        """Check if we should reset counter due to timeout."""
        if self.last_detection_time > 0:
            time_since_last = time.time() - self.last_detection_time
            if time_since_last > self.reset_timeout:
                self._reset_counter()
                self.last_detection_time = 0

    def _trigger_webhook(self):
        """Trigger the Home Assistant webhook."""
        print("\n" + "=" * 60, flush=True)
        print(f"  {self.consecutive_required} CONSECUTIVE CLICKS! "
              f"Triggering webhook...", flush=True)
        print(f"  Trigger #{self.total_triggers + 1}", flush=True)
        print("=" * 60, flush=True)

        try:
            response = requests.post(self.webhook_url, timeout=10)
            if response.status_code == 200:
                print("  Webhook triggered successfully!", flush=True)
                self.total_triggers += 1
            else:
                print(f"  Failed (status: {response.status_code})", flush=True)
        except Exception as e:
            print(f"  Webhook error: {e}", flush=True)

        print("=" * 60 + "\n", flush=True)

    def _save_audio(self, audio, confidence):
        """Save detected click audio to file."""
        if not self.save_clicks:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"click_{self.total_clicks:04d}_{timestamp}_conf{confidence * 100:.0f}.wav"
        filepath = self.save_dir / filename
        sf.write(str(filepath), audio, self.sample_rate)
        return filename

    def _on_click_detected(self, audio_chunk, confidence):
        """Handle a detected click with debouncing and consecutive logic."""
        current_time = time.time()

        # Debounce
        if current_time - self.last_click_time < self.debounce_interval:
            return

        self.last_click_time = current_time
        self.last_detection_time = current_time
        self.consecutive_count += 1
        self.total_clicks += 1

        # Save audio
        filename = self._save_audio(audio_chunk, confidence)
        save_info = f" | Saved: {filename}" if filename else ""

        print(f"  CLICK {self.consecutive_count}/{self.consecutive_required} | "
              f"Confidence: {confidence:.1%} | "
              f"Total: {self.total_clicks}{save_info}", flush=True)

        # Check if we hit the threshold
        if self.consecutive_count >= self.consecutive_required:
            self._trigger_webhook()
            self.consecutive_count = 0
            self.last_detection_time = 0
            print("  Counter reset. Listening for next sequence...\n", flush=True)

    def _audio_callback(self, indata, frames, time_info, status):
        """Quickly copy audio to queue -- no heavy processing here."""
        if status:
            print(f"  Stream status: {status}", flush=True)
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _processing_thread(self):
        """Process audio chunks from queue in a separate thread."""
        gc_counter = 0

        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                self._check_timeout()
                continue

            self._check_timeout()

            is_click, confidence = self.detect(audio_chunk)

            if is_click:
                self._on_click_detected(audio_chunk, confidence)
            else:
                self.filtered_count += 1

            gc_counter += 1
            if gc_counter >= 600:
                gc_counter = 0
                gc.collect()

    def run(self):
        """Start continuous listening with auto-restart."""
        print("\n" + "=" * 60)
        print("CALL FOR ATTENTION - Tongue Click Detector")
        print("=" * 60)
        print(f"Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sample rate  : {self.sample_rate} Hz")
        print(f"Confidence   : {self.confidence_threshold:.0%}")
        print(f"Consecutive  : {self.consecutive_required} clicks to trigger")
        print(f"Timeout      : {self.reset_timeout}s (counter resets)")
        print(f"Debounce     : {self.debounce_interval}s between clicks")
        print(f"Webhook      : {self.webhook_url}")
        if self.save_clicks:
            print(f"Saving clips : {self.save_dir}/")
        if self.device is not None:
            print(f"Device       : {self.device}")
        print("=" * 60)
        print("\nListening... make tongue clicks!")
        print("Press Ctrl+C to stop\n", flush=True)

        self.start_time = time.time()

        # Start processing thread
        processor = threading.Thread(target=self._processing_thread, daemon=True)
        processor.start()

        try:
            with sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                device=self.device
            ):
                while self.running:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("\n\nStopped by user")
            self.running = False
        except Exception as e:
            if self.running:
                self.restart_count += 1
                wait_time = min(5 * (2 ** min(self.restart_count - 1, 3)), 60)
                print(f"\nError: {e}")
                print(f"Restart #{self.restart_count}, waiting {wait_time}s...")
                time.sleep(wait_time)

                if self.running:
                    print("Restarting listener...\n")
                    self.stop_event.clear()
                    return self.run()
        finally:
            self.stop_event.set()
            processor.join(timeout=2)
            if not self.running:
                self._print_summary()

    def _print_summary(self):
        """Print session summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Runtime           : {hours}h {minutes}m {seconds}s")
        print(f"Clicks detected   : {self.total_clicks}")
        print(f"Webhooks triggered: {self.total_triggers}")
        print(f"Counter at stop   : {self.consecutive_count}/{self.consecutive_required}")
        print(f"Chunks filtered   : {self.filtered_count}")
        print(f"Restarts          : {self.restart_count}")
        if self.save_clicks:
            saved_files = len(list(self.save_dir.glob('*.wav')))
            print(f"Audio files saved : {saved_files}")
            print(f"Saved to          : {self.save_dir}/")
        print(f"Ended             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Tongue click detector that triggers Home Assistant webhook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (3 consecutive clicks, MacBook mic)
  python call_for_attention.py

  # Stricter confidence
  python call_for_attention.py --threshold 0.85

  # Require 5 consecutive clicks
  python call_for_attention.py --consecutive 5

  # Use Jabra mic at 16kHz with 16k model
  python call_for_attention.py --device 2 --sample-rate 16000 --model-dir models_16k

  # Custom webhook URL
  python call_for_attention.py --webhook-url https://your-ha/api/webhook/hook_id

  # Disable saving click audio files
  python call_for_attention.py --no-save

  # List audio devices
  python call_for_attention.py --list-devices
        """
    )

    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold 0-1 (default: 0.7)')
    parser.add_argument('--consecutive', type=int, default=3,
                        help='Consecutive clicks to trigger webhook (default: 3)')
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='Seconds without click before counter resets (default: 2.0)')
    parser.add_argument('--debounce', type=float, default=0.5,
                        help='Minimum seconds between clicks (default: 0.5)')
    parser.add_argument('--webhook-url', type=str,
                        default='https://ut-beachhome.homeadapt.us/api/webhook/tongue_click_alert',
                        help='Home Assistant webhook URL')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing model files (default: models)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate (default: 44100, Jabra: 16000)')
    parser.add_argument('--min-energy', type=float, default=0.02,
                        help='Minimum energy threshold (default: 0.02)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save detected click audio files')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')

    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        return

    try:
        listener = CallForAttention(
            model_path=f"{args.model_dir}/tongue_click_model.pkl",
            scaler_path=f"{args.model_dir}/scaler.pkl",
            sample_rate=args.sample_rate,
            confidence_threshold=args.threshold,
            min_energy=args.min_energy,
            webhook_url=args.webhook_url,
            consecutive_required=args.consecutive,
            reset_timeout=args.timeout,
            debounce_interval=args.debounce,
            save_clicks=not args.no_save,
            device=args.device,
        )
        listener.run()

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python retrain_model.py --positive training_data/positives --negative training_data/auto_collected")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
