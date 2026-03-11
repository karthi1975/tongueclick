#!/usr/bin/env python3
"""
Call For Attention - Tongue Click Webhook Trigger

Continuously listens for tongue clicks using the trained scikit-learn
Random Forest model. After detecting consecutive clicks, triggers a
Home Assistant webhook to fire an alarm/bell.

Usage:
    # Run with defaults (3-pause-3 pattern, 0.93 confidence)
    python call_for_attention.py

    # Use Jabra mic at 16kHz with 16k model
    python call_for_attention.py --device 2 --sample-rate 16000 --model-dir models_16k

    # Custom pattern: 4 clicks, pause, 4 clicks
    python call_for_attention.py --clicks-per-group 4

    # Adjust pause window
    python call_for_attention.py --pause-min 0.5 --pause-max 3.0

    # List audio devices
    python call_for_attention.py --list-devices
"""

import warnings
warnings.filterwarnings("ignore")

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

    # Confirmation pattern states
    STATE_WAITING_FIRST_GROUP = 'waiting_first_group'
    STATE_WAITING_PAUSE = 'waiting_pause'
    STATE_WAITING_SECOND_GROUP = 'waiting_second_group'

    def __init__(self, model_path='models/tongue_click_model.pkl',
                 scaler_path='models/scaler.pkl',
                 sample_rate=44100,
                 confidence_threshold=0.90,
                 min_energy=0.02,
                 webhook_url='https://ut-beachhome.homeadapt.us/api/webhook/tongue_click_alert',
                 clicks_per_group=3,
                 group_timeout=3.0,
                 pause_min=0.2,
                 pause_max=5.0,
                 rhythm_max_cv=0.8,
                 debounce_interval=0.15,
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
            clicks_per_group: Number of clicks in each group of the pattern
            group_timeout: Seconds without click before a group resets
            pause_min: Minimum pause between groups (seconds)
            pause_max: Maximum pause between groups (seconds)
            rhythm_max_cv: Maximum coefficient of variation for rhythm check (0-1)
            debounce_interval: Minimum seconds between click detections
            save_clicks: Whether to save detected click audio files
            device: Audio input device index
        """
        self.sample_rate = sample_rate
        self.model_sample_rate = 44100
        self.confidence_threshold = confidence_threshold
        self.min_energy = min_energy
        self.webhook_url = webhook_url
        self.clicks_per_group = clicks_per_group
        self.group_timeout = group_timeout
        self.pause_min = pause_min
        self.pause_max = pause_max
        self.rhythm_max_cv = rhythm_max_cv
        self.debounce_interval = debounce_interval
        self.save_clicks = save_clicks
        self.device = device

        # Load model and scaler
        print("Loading ML model...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        # Force single-threaded prediction (prevents warning spam on Pi)
        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = 1
        print(f"  Model loaded from: {model_path}")

        # Feature extractor at model's training sample rate
        self.feature_extractor = AdvancedFeatureExtractor(self.model_sample_rate)

        # State - confirmation pattern: [3 clicks] [pause] [3 clicks]
        self.state = self.STATE_WAITING_FIRST_GROUP
        self.click_times = []  # timestamps of clicks in current group
        self.first_group_end_time = 0  # when first group completed
        self.total_clicks = 0
        self.total_triggers = 0
        self.filtered_count = 0
        self.false_rhythm_count = 0
        self.last_click_time = 0
        self.start_time = None
        self.restart_count = 0
        self.overflow_count = 0
        self.needs_restart = False
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
            self.last_cleanup_time = time.time()
            self.cleanup_interval = 300  # 5 minutes

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

    def _reset_state(self, reason=""):
        """Reset the pattern state machine."""
        old_state = self.state
        self.state = self.STATE_WAITING_FIRST_GROUP
        self.click_times = []
        self.first_group_end_time = 0
        if reason:
            print(f"  RESET ({reason}) - was in {old_state}", flush=True)

    def _check_rhythm(self, click_times):
        """
        Check if click timestamps are rhythmically regular.
        Returns True if clicks are evenly spaced (low coefficient of variation).
        """
        if len(click_times) < 2:
            return True
        intervals = [click_times[i+1] - click_times[i]
                     for i in range(len(click_times) - 1)]
        mean_interval = np.mean(intervals)
        if mean_interval < 0.05:
            return False  # Too fast, likely noise
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval
        return cv <= self.rhythm_max_cv

    def _check_timeout(self):
        """Check for timeouts based on current state."""
        now = time.time()

        if self.state == self.STATE_WAITING_FIRST_GROUP:
            # Reset if too long since last click in group
            if self.click_times and (now - self.click_times[-1]) > self.group_timeout:
                self._reset_state("group 1 timeout")

        elif self.state == self.STATE_WAITING_PAUSE:
            # If pause is too long, reset everything
            if (now - self.first_group_end_time) > self.pause_max:
                self._reset_state("pause too long")

        elif self.state == self.STATE_WAITING_SECOND_GROUP:
            # Reset if too long since last click in second group
            if self.click_times and (now - self.click_times[-1]) > self.group_timeout:
                self._reset_state("group 2 timeout")

    def _trigger_webhook(self):
        """Trigger the Home Assistant webhook."""
        print("\n" + "*" * 60, flush=True)
        print("*" * 60, flush=True)
        print(f"  RHYTHM RECOGNIZED! "
              f"({self.clicks_per_group} clicks - pause - "
              f"{self.clicks_per_group} clicks)", flush=True)
        print(f"  >>> This WILL fire the webhook when enabled <<<", flush=True)
        print(f"  Trigger #{self.total_triggers + 1} | "
              f"{datetime.now().strftime('%H:%M:%S')}", flush=True)
        print("*" * 60, flush=True)
        print("*" * 60, flush=True)

        try:
            response = requests.post(self.webhook_url, timeout=10)
            if response.status_code == 200:
                print("  Webhook triggered successfully!", flush=True)
                self.total_triggers += 1
            else:
                print(f"  Failed (status: {response.status_code})", flush=True)
        except Exception as e:
            print(f"  Webhook error: {e}", flush=True)

        print("*" * 60 + "\n", flush=True)

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
        """Handle a detected click with pattern confirmation logic.

        Pattern: [N clicks] [pause 0.8-2.5s] [N clicks] -> trigger
        Each group must be rhythmically regular.
        """
        current_time = time.time()

        # Debounce
        if current_time - self.last_click_time < self.debounce_interval:
            return

        self.last_click_time = current_time
        self.total_clicks += 1

        # Save audio
        filename = self._save_audio(audio_chunk, confidence)
        save_info = f" | Saved: {filename}" if filename else ""

        if self.state == self.STATE_WAITING_FIRST_GROUP:
            self.click_times.append(current_time)
            count = len(self.click_times)
            print(f"  GROUP 1: click {count}/{self.clicks_per_group} | "
                  f"Conf: {confidence:.1%}{save_info}", flush=True)

            if count >= self.clicks_per_group:
                # Check rhythm of first group
                if self._check_rhythm(self.click_times):
                    self.first_group_end_time = current_time
                    self.state = self.STATE_WAITING_PAUSE
                    self.click_times = []
                    print(f"  >> Group 1 complete! Now PAUSE for "
                          f"{self.pause_min}-{self.pause_max}s, "
                          f"then {self.clicks_per_group} more clicks...",
                          flush=True)
                else:
                    self.false_rhythm_count += 1
                    self._reset_state("irregular rhythm in group 1")

        elif self.state == self.STATE_WAITING_PAUSE:
            # A click arrived during expected pause period
            pause_duration = current_time - self.first_group_end_time
            if pause_duration < self.pause_min:
                # Pause was too short - this click is too early
                # Treat as continuation / noise, reset
                self._reset_state(f"pause too short ({pause_duration:.1f}s)")
            else:
                # Pause was valid, this click starts group 2
                self.state = self.STATE_WAITING_SECOND_GROUP
                self.click_times = [current_time]
                print(f"  GROUP 2: click 1/{self.clicks_per_group} | "
                      f"Conf: {confidence:.1%} | "
                      f"Pause was {pause_duration:.1f}s{save_info}",
                      flush=True)

        elif self.state == self.STATE_WAITING_SECOND_GROUP:
            self.click_times.append(current_time)
            count = len(self.click_times)
            print(f"  GROUP 2: click {count}/{self.clicks_per_group} | "
                  f"Conf: {confidence:.1%}{save_info}", flush=True)

            if count >= self.clicks_per_group:
                # Check rhythm of second group
                if self._check_rhythm(self.click_times):
                    self._trigger_webhook()
                    self._reset_state("")
                    print("  Pattern reset. Listening for next sequence...\n",
                          flush=True)
                else:
                    self.false_rhythm_count += 1
                    self._reset_state("irregular rhythm in group 2")

    def _audio_callback(self, indata, frames, time_info, status):
        """Quickly copy audio to queue -- no heavy processing here."""
        if status:
            status_str = str(status)
            print(f"  Stream status: {status}", flush=True)
            if 'overflow' in status_str.lower():
                self.overflow_count += 1
                if self.overflow_count >= 10:
                    print("  [WARNING] Too many overflows, will restart...",
                          flush=True)
                    self.needs_restart = True
            else:
                self.overflow_count = 0
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
                self._cleanup_old_clicks()

    def _cleanup_old_clicks(self):
        """Delete saved click files older than 5 minutes."""
        if not self.save_clicks:
            return
        now = time.time()
        if now - self.last_cleanup_time < self.cleanup_interval:
            return
        self.last_cleanup_time = now
        count = 0
        for f in self.save_dir.glob("click_*.wav"):
            if now - f.stat().st_mtime > self.cleanup_interval:
                f.unlink()
                count += 1
        if count > 0:
            print(f"  [Cleanup] Deleted {count} old click file(s)", flush=True)

    def run(self):
        """Start continuous listening with auto-restart."""
        print("\n" + "=" * 60)
        print("CALL FOR ATTENTION - Tongue Click Detector")
        print("=" * 60)
        print(f"Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sample rate  : {self.sample_rate} Hz")
        print(f"Confidence   : {self.confidence_threshold:.0%}")
        print(f"Pattern      : {self.clicks_per_group} clicks, "
              f"pause {self.pause_min}-{self.pause_max}s, "
              f"{self.clicks_per_group} clicks")
        print(f"Rhythm check : CV <= {self.rhythm_max_cv}")
        print(f"Group timeout: {self.group_timeout}s")
        print(f"Debounce     : {self.debounce_interval}s between clicks")
        print(f"Webhook      : {self.webhook_url}")
        if self.save_clicks:
            print(f"Saving clips : {self.save_dir}/")
        if self.device is not None:
            print(f"Device       : {self.device}")
        print("=" * 60)
        print(f"\nListening... pattern: "
              f"{self.clicks_per_group} clicks -> pause -> "
              f"{self.clicks_per_group} clicks")
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
                    if self.needs_restart:
                        print("\n  [AUTO-RESTART] Restarting due to overflow...\n",
                              flush=True)
                        self.needs_restart = False
                        self.overflow_count = 0
                        self._reset_state("")
                        break

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

        # Auto-restart after overflow break
        if self.running:
            self.restart_count += 1
            self.stop_event.clear()
            return self.run()

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
        print(f"State at stop     : {self.state} ({len(self.click_times)} clicks)")
        print(f"Rhythm rejects    : {self.false_rhythm_count}")
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
  # Run with defaults (3-pause-3 pattern, 0.93 confidence)
  python call_for_attention.py

  # Use Jabra mic at 16kHz with 16k model
  python call_for_attention.py --device 2 --sample-rate 16000 --model-dir models_16k

  # Custom pattern: 2 clicks, pause, 2 clicks (easier for user)
  python call_for_attention.py --clicks-per-group 2

  # Stricter rhythm and threshold
  python call_for_attention.py --threshold 0.95 --rhythm-max-cv 0.3

  # Disable saving click audio files
  python call_for_attention.py --no-save

  # List audio devices
  python call_for_attention.py --list-devices
        """
    )

    parser.add_argument('--threshold', type=float, default=0.90,
                        help='Confidence threshold 0-1 (default: 0.90)')
    parser.add_argument('--clicks-per-group', type=int, default=3,
                        help='Clicks per group in the pattern (default: 3)')
    parser.add_argument('--group-timeout', type=float, default=3.0,
                        help='Seconds without click before group resets (default: 3.0)')
    parser.add_argument('--pause-min', type=float, default=0.2,
                        help='Minimum pause between groups (default: 0.2s)')
    parser.add_argument('--pause-max', type=float, default=5.0,
                        help='Maximum pause between groups (default: 5.0s)')
    parser.add_argument('--rhythm-max-cv', type=float, default=0.8,
                        help='Max coefficient of variation for rhythm (default: 0.8)')
    parser.add_argument('--debounce', type=float, default=0.15,
                        help='Minimum seconds between clicks (default: 0.15)')
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
            clicks_per_group=args.clicks_per_group,
            group_timeout=args.group_timeout,
            pause_min=args.pause_min,
            pause_max=args.pause_max,
            rhythm_max_cv=args.rhythm_max_cv,
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
