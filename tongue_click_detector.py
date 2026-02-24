"""
Tongue Click Sound Recognition System (Refactored with SOLID Principles)

This module provides a modular, extensible system for detecting tongue click sounds
in real-time audio streams or pre-recorded audio files.

SOLID Principles Applied:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Easy to extend with new detectors/sources without modification
- Liskov Substitution: Abstract base classes ensure proper substitution
- Interface Segregation: Focused interfaces for specific needs
- Dependency Inversion: Depends on abstractions, uses dependency injection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np
import sounddevice as sd
import librosa
import time

# Pre-import librosa submodules to avoid lazy-loading during audio callback
# This prevents OSError from llvmlite during real-time processing
import librosa.onset
import librosa.feature
import librosa.filters


# ============================================================================
# VALUE OBJECTS & DATA CLASSES
# ============================================================================

@dataclass
class AudioFeatures:
    """Encapsulates extracted audio features."""
    onset_strength: float
    spectral_centroid: float
    rms_energy_peak: float
    rms_energy_mean: float
    zero_crossing_rate: float

    def is_impulsive(self, min_ratio: float = 3.0) -> bool:
        """Check if audio shows impulsive characteristics."""
        return self.rms_energy_peak > self.rms_energy_mean * min_ratio

    def has_high_frequency(self, min_freq: float = 2000) -> bool:
        """Check if audio has high frequency content (typical for clicks)."""
        return self.spectral_centroid > min_freq

    @property
    def peak_to_mean_ratio(self) -> float:
        """Get the peak-to-mean energy ratio."""
        return self.rms_energy_peak / (self.rms_energy_mean + 1e-8)


@dataclass
class ClickDetectionResult:
    """Result of click detection analysis."""
    is_click: bool
    confidence_score: float
    timestamp: Optional[float] = None
    features: Optional[AudioFeatures] = None


@dataclass
class DetectorConfig:
    """Configuration for the click detector."""
    sample_rate: int = 44100
    threshold: float = 8.0  # ONSET strength (not 0-1, actual value!): Clicks=10-25, Speech=0-7
    onset_weight: float = 0.5  # Onset is most important
    frequency_weight: float = 0.3
    impulsive_weight: float = 0.2
    confidence_threshold: float = 0.65  # Balanced
    min_energy_threshold: float = 0.01  # Ignore very quiet sounds
    chunk_duration: float = 0.1  # seconds
    min_spectral_centroid: float = 2200  # Clicks=2000-4300Hz, Speech=900-2400Hz
    min_peak_to_mean_ratio: float = 1.6  # Clicks=1.6-2.4x, Speech=1.1-2.0x


# ============================================================================
# INTERFACES (Abstract Base Classes)
# ============================================================================

class IAudioFeatureExtractor(ABC):
    """Interface for audio feature extraction (ISP - Interface Segregation)."""

    @abstractmethod
    def extract_features(self, audio_chunk: np.ndarray) -> AudioFeatures:
        """Extract relevant features from audio chunk."""
        pass


class IClickClassifier(ABC):
    """Interface for click classification logic (ISP)."""

    @abstractmethod
    def classify(self, features: AudioFeatures) -> ClickDetectionResult:
        """Classify whether features indicate a click sound."""
        pass


class IAudioSource(ABC):
    """Interface for audio sources (OCP - Open for extension)."""

    @abstractmethod
    def process_audio(self, detector: 'IClickDetector',
                     callback: Optional[Callable] = None) -> List[ClickDetectionResult]:
        """Process audio from this source."""
        pass


class IClickDetector(ABC):
    """Interface for the main click detector (DIP - Dependency Inversion)."""

    @abstractmethod
    def detect(self, audio_chunk: np.ndarray) -> ClickDetectionResult:
        """Detect click in audio chunk."""
        pass


class IEventHandler(ABC):
    """Interface for handling detection events (ISP)."""

    @abstractmethod
    def on_click_detected(self, result: ClickDetectionResult) -> None:
        """Handle click detection event."""
        pass


# ============================================================================
# CONCRETE IMPLEMENTATIONS - FEATURE EXTRACTION (SRP)
# ============================================================================

class LibrosaFeatureExtractor(IAudioFeatureExtractor):
    """
    Extracts audio features using librosa library.
    Single Responsibility: Audio feature extraction only.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def extract_features(self, audio_chunk: np.ndarray) -> AudioFeatures:
        """
        Extract features relevant for tongue click detection:
        - Onset strength (sharp transients)
        - Spectral centroid (frequency content)
        - RMS energy (amplitude envelope)
        - Zero-crossing rate (signal characteristics)
        """
        # Normalize audio to prevent numerical issues
        normalized_audio = self._normalize(audio_chunk)

        # Extract features
        onset_strength = self._extract_onset_strength(normalized_audio)
        spectral_centroid = self._extract_spectral_centroid(normalized_audio)
        rms_peak, rms_mean = self._extract_rms_energy(normalized_audio)
        zcr = self._extract_zero_crossing_rate(normalized_audio)

        return AudioFeatures(
            onset_strength=onset_strength,
            spectral_centroid=spectral_centroid,
            rms_energy_peak=rms_peak,
            rms_energy_mean=rms_mean,
            zero_crossing_rate=zcr
        )

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        return audio / (max_val + 1e-8) if max_val > 0 else audio

    def _extract_onset_strength(self, audio: np.ndarray) -> float:
        """Extract maximum onset strength."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        return float(np.max(onset_env))

    def _extract_spectral_centroid(self, audio: np.ndarray) -> float:
        """Extract mean spectral centroid."""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        return float(np.mean(centroid))

    def _extract_rms_energy(self, audio: np.ndarray) -> Tuple[float, float]:
        """Extract RMS energy peak and mean."""
        rms = librosa.feature.rms(y=audio)[0]
        return float(np.max(rms)), float(np.mean(rms))

    def _extract_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Extract mean zero-crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        return float(np.mean(zcr))


# ============================================================================
# CONCRETE IMPLEMENTATIONS - CLASSIFICATION (SRP)
# ============================================================================

class WeightedFeatureClassifier(IClickClassifier):
    """
    Classifies clicks using weighted feature combination.
    Single Responsibility: Click classification logic only.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

    def classify(self, features: AudioFeatures) -> ClickDetectionResult:
        """
        Classify click using weighted feature scores.

        Tongue clicks are characterized by:
        - Very sharp onset (sudden amplitude increase)
        - Very high frequency content (broadband energy > 3kHz)
        - Very impulsive nature (high peak-to-mean energy ratio > 4x)

        Speech is filtered out because:
        - Speech has lower frequency content (mostly 100-3000 Hz)
        - Speech has longer duration and smoother envelope
        - Speech has lower peak-to-mean ratio
        """
        # Evaluate individual features with STRICT thresholds
        has_sharp_onset = features.onset_strength > self.config.threshold
        has_high_freq = features.has_high_frequency(self.config.min_spectral_centroid)
        is_impulsive = features.is_impulsive(self.config.min_peak_to_mean_ratio)

        # Additional filtering: Check peak-to-mean ratio
        peak_ratio = features.peak_to_mean_ratio
        meets_impulsive_threshold = peak_ratio >= self.config.min_peak_to_mean_ratio

        # Calculate weighted confidence score
        confidence_score = (
            has_sharp_onset * self.config.onset_weight +
            has_high_freq * self.config.frequency_weight +
            is_impulsive * self.config.impulsive_weight
        )

        # STRICT DETECTION: Must meet ALL criteria AND pass confidence threshold
        is_click = (
            has_sharp_onset and
            has_high_freq and
            meets_impulsive_threshold and
            confidence_score > self.config.confidence_threshold
        )

        return ClickDetectionResult(
            is_click=is_click,
            confidence_score=confidence_score,
            features=features
        )


# ============================================================================
# CONCRETE IMPLEMENTATIONS - MAIN DETECTOR (SRP + DIP)
# ============================================================================

class TongueClickDetector(IClickDetector):
    """
    Main click detector using composition and dependency injection.
    Single Responsibility: Orchestrate feature extraction and classification.
    Dependency Inversion: Depends on abstractions (interfaces), not concrete classes.
    """

    def __init__(self,
                 feature_extractor: IAudioFeatureExtractor,
                 classifier: IClickClassifier,
                 config: DetectorConfig):
        """
        Initialize detector with injected dependencies.

        Args:
            feature_extractor: Component to extract audio features
            classifier: Component to classify clicks
            config: Detector configuration
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.config = config

    def detect(self, audio_chunk: np.ndarray) -> ClickDetectionResult:
        """
        Detect click in audio chunk.

        Returns:
            ClickDetectionResult with detection status and confidence
        """
        # Extract features
        features = self.feature_extractor.extract_features(audio_chunk)

        # Classify
        result = self.classifier.classify(features)

        return result

    def has_sufficient_energy(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk has sufficient energy for analysis."""
        return np.max(np.abs(audio_chunk)) > self.config.min_energy_threshold


# ============================================================================
# CONCRETE IMPLEMENTATIONS - AUDIO SOURCES (SRP + OCP)
# ============================================================================

class RealTimeAudioSource(IAudioSource):
    """
    Real-time audio stream source.
    Single Responsibility: Handle real-time audio streaming.
    Open/Closed: Can extend with different streaming strategies.
    Memory-safe for long-running sessions (24h+).
    """

    # Keep only the most recent N detections in memory
    MAX_RESULTS_IN_MEMORY = 10000

    def __init__(self, config: DetectorConfig, duration: int = 10):
        self.config = config
        self.duration = duration
        self.start_time = None
        self.results: List[ClickDetectionResult] = []
        self.total_click_count = 0

    def process_audio(self, detector: IClickDetector,
                     callback: Optional[Callable] = None) -> List[ClickDetectionResult]:
        """Process real-time audio stream."""
        import gc

        print(f"Listening for tongue clicks for {self.duration} seconds...")
        print("Make tongue click sounds!")

        self.start_time = time.time()
        self.results = []
        self.total_click_count = 0
        gc_counter = [0]  # Mutable counter for use inside callback

        chunk_samples = int(self.config.sample_rate * self.config.chunk_duration)

        def audio_callback(indata, frames, time_info, status):
            """Process incoming audio data."""
            _ = frames  # Unused but required by sounddevice API
            _ = time_info  # Unused but required by sounddevice API

            if status:
                print(f"Stream status: {status}")

            audio_chunk = indata[:, 0]  # Get mono channel

            # Only process if sufficient energy
            if detector.has_sufficient_energy(audio_chunk):
                result = detector.detect(audio_chunk)

                if result.is_click:
                    result.timestamp = time.time() - self.start_time
                    self.results.append(result)
                    self.total_click_count += 1
                    print(f"✓ Click detected! (confidence: {result.confidence_score:.2f}) "
                          f"at {result.timestamp:.2f}s")

                    if callback:
                        callback(result)

                    # Trim results list to prevent unbounded memory growth
                    if len(self.results) > self.MAX_RESULTS_IN_MEMORY:
                        self.results = self.results[-self.MAX_RESULTS_IN_MEMORY // 2:]

            # Periodic GC every ~60 seconds (600 chunks at 100ms each)
            gc_counter[0] += 1
            if gc_counter[0] >= 600:
                gc_counter[0] = 0
                gc.collect()

        try:
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.config.sample_rate,
                              blocksize=chunk_samples):
                sd.sleep(int(self.duration * 1000))
        except KeyboardInterrupt:
            print("\nStopped by user")

        print(f"\nTotal clicks detected: {self.total_click_count}")
        return self.results


class AudioFileSource(IAudioSource):
    """
    Pre-recorded audio file source.
    Single Responsibility: Handle file-based audio analysis.
    Open/Closed: Can extend with different file formats.
    """

    def __init__(self, filepath: str, config: DetectorConfig):
        self.filepath = filepath
        self.config = config

    def process_audio(self, detector: IClickDetector,
                     callback: Optional[Callable] = None) -> List[ClickDetectionResult]:
        """Process audio from file."""
        print(f"Analyzing audio file: {self.filepath}")

        try:
            # Load audio file
            audio, sr = librosa.load(self.filepath, sr=self.config.sample_rate)
            print(f"Loaded audio: {len(audio)/sr:.2f} seconds")

            results = self._analyze_audio(audio, detector, callback)

            print(f"\nTotal clicks detected: {len(results)}")
            return results

        except Exception as e:
            print(f"Error analyzing file: {e}")
            return []

    def _analyze_audio(self, audio: np.ndarray, detector: IClickDetector,
                      callback: Optional[Callable]) -> List[ClickDetectionResult]:
        """Analyze audio signal for clicks."""
        chunk_length = int(self.config.chunk_duration * self.config.sample_rate)
        results = []

        # Use overlapping windows (50% overlap)
        hop_length = chunk_length // 2

        for i in range(0, len(audio) - chunk_length, hop_length):
            chunk = audio[i:i + chunk_length]

            if detector.has_sufficient_energy(chunk):
                result = detector.detect(chunk)

                if result.is_click:
                    result.timestamp = i / self.config.sample_rate
                    results.append(result)
                    print(f"Click at {result.timestamp:.2f}s "
                          f"(confidence: {result.confidence_score:.2f})")

                    if callback:
                        callback(result)

        return results


# ============================================================================
# EVENT HANDLERS (SRP + ISP)
# ============================================================================

class ConsoleEventHandler(IEventHandler):
    """
    Console-based event handler.
    Single Responsibility: Handle console output for events.
    """

    def on_click_detected(self, result: ClickDetectionResult) -> None:
        """Print click detection to console."""
        timestamp_str = f" at {result.timestamp:.2f}s" if result.timestamp else ""
        print(f"✓ Click detected{timestamp_str} (confidence: {result.confidence_score:.2f})")


class LoggingEventHandler(IEventHandler):
    """
    Logging-based event handler.
    Single Responsibility: Handle logging for events.
    """

    def __init__(self):
        self.log: List[ClickDetectionResult] = []

    def on_click_detected(self, result: ClickDetectionResult) -> None:
        """Log click detection."""
        self.log.append(result)


class CompositeEventHandler(IEventHandler):
    """
    Composite event handler supporting multiple handlers.
    Follows Composite pattern for extensibility.
    """

    def __init__(self, handlers: List[IEventHandler]):
        self.handlers = handlers

    def on_click_detected(self, result: ClickDetectionResult) -> None:
        """Notify all handlers."""
        for handler in self.handlers:
            handler.on_click_detected(result)


# ============================================================================
# FACTORY (Simplifies object creation)
# ============================================================================

class DetectorFactory:
    """
    Factory for creating configured detector instances.
    Simplifies dependency injection setup.
    """

    @staticmethod
    def create_default_detector(config: Optional[DetectorConfig] = None) -> TongueClickDetector:
        """Create a detector with default configuration."""
        if config is None:
            config = DetectorConfig()

        feature_extractor = LibrosaFeatureExtractor(sample_rate=config.sample_rate)
        classifier = WeightedFeatureClassifier(config=config)

        return TongueClickDetector(
            feature_extractor=feature_extractor,
            classifier=classifier,
            config=config
        )

    @staticmethod
    def create_real_time_source(config: DetectorConfig, duration: int = 10) -> RealTimeAudioSource:
        """Create real-time audio source."""
        return RealTimeAudioSource(config=config, duration=duration)

    @staticmethod
    def create_file_source(filepath: str, config: DetectorConfig) -> AudioFileSource:
        """Create file-based audio source."""
        return AudioFileSource(filepath=filepath, config=config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_audio_devices():
    """List available audio input devices."""
    print("Available audio devices:")
    print(sd.query_devices())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tongue Click Detector - Real-time tongue click detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tongue_click_detector.py                          # Default settings
  python tongue_click_detector.py --threshold 6.0          # More sensitive
  python tongue_click_detector.py --threshold 12.0         # Stricter (noisy env)
  python tongue_click_detector.py --duration 30            # Listen for 30 seconds
  python tongue_click_detector.py --confidence 0.75        # Higher confidence
  python tongue_click_detector.py --file recording.wav     # Analyze a file

Presets:
  Sensitive:  --threshold 6.0 --confidence 0.55 --centroid 2000 --peak-ratio 1.5
  Default:    --threshold 8.0 --confidence 0.65 --centroid 2200 --peak-ratio 1.6
  Strict:     --threshold 12.0 --confidence 0.75 --centroid 2500 --peak-ratio 1.8
        """
    )
    parser.add_argument('--threshold', type=float, default=8.0,
                        help='Onset detection threshold (default: 8.0, lower=more sensitive)')
    parser.add_argument('--confidence', type=float, default=0.65,
                        help='Confidence threshold 0-1 (default: 0.65)')
    parser.add_argument('--centroid', type=float, default=2200,
                        help='Min spectral centroid in Hz (default: 2200)')
    parser.add_argument('--peak-ratio', type=float, default=1.6,
                        help='Min peak-to-mean ratio (default: 1.6)')
    parser.add_argument('--energy', type=float, default=0.01,
                        help='Min energy threshold (default: 0.01)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Listening duration in seconds (default: 10)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate in Hz (default: 44100)')
    parser.add_argument('--file', type=str, default=None,
                        help='Analyze an audio file instead of real-time')

    args = parser.parse_args()

    config = DetectorConfig(
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        min_spectral_centroid=args.centroid,
        min_peak_to_mean_ratio=args.peak_ratio,
        min_energy_threshold=args.energy,
    )

    # Create detector using factory (Dependency Injection)
    detector = DetectorFactory.create_default_detector(config)

    # List available audio devices
    print("=" * 60)
    list_audio_devices()
    print("=" * 60)

    if args.file:
        # File analysis mode
        print(f"\nAnalyzing file: {args.file}")
        file_source = DetectorFactory.create_file_source(args.file, config)
        event_handler = ConsoleEventHandler()
        results = file_source.process_audio(
            detector=detector,
            callback=lambda result: event_handler.on_click_detected(result)
        )
    else:
        # Real-time detection mode
        print(f"\nStarting real-time detection (duration: {args.duration}s, threshold: {args.threshold})...")
        print("Press Ctrl+C to stop early")

        audio_source = DetectorFactory.create_real_time_source(config, duration=args.duration)
        event_handler = ConsoleEventHandler()
        results = audio_source.process_audio(
            detector=detector,
            callback=lambda result: event_handler.on_click_detected(result)
        )
