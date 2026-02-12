#!/usr/bin/env python3
"""
Advanced Feature Extraction for Tongue Click Detection
Includes features to distinguish clicks from household sounds
"""

import numpy as np
import librosa
from scipy import signal, stats
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class AdvancedAudioFeatures:
    """Comprehensive feature set for robust click detection."""

    # Temporal features
    duration_ms: float
    attack_time_ms: float
    decay_time_ms: float
    kurtosis: float

    # Spectral features
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_flatness: float
    spectral_flux: float
    spectral_centroid_variance: float

    # Energy features
    rms_peak: float
    rms_mean: float
    peak_to_mean_ratio: float
    onset_strength: float

    # Harmonic features
    harmonic_to_noise_ratio: float
    has_pitch: bool
    zero_crossing_rate: float

    # Frequency distribution
    low_freq_ratio: float  # Energy below 1kHz / total
    mid_freq_ratio: float  # Energy 1-3kHz / total
    high_freq_ratio: float  # Energy above 3kHz / total

    # Decay characteristics
    has_ringing: bool
    decay_type: str  # 'fast', 'slow', 'ringing'

    # MFCCs (timbre)
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    mfcc_5: float


class AdvancedFeatureExtractor:
    """Extract comprehensive features for distinguishing tongue clicks."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def extract_all_features(self, audio_chunk: np.ndarray) -> AdvancedAudioFeatures:
        """Extract all features from audio chunk."""

        # Normalize
        audio_norm = self._normalize(audio_chunk)

        # Extract all feature categories
        temporal_features = self._extract_temporal_features(audio_norm)
        spectral_features = self._extract_spectral_features(audio_norm)
        energy_features = self._extract_energy_features(audio_norm)
        harmonic_features = self._extract_harmonic_features(audio_norm)
        freq_dist = self._extract_frequency_distribution(audio_norm)
        decay_features = self._extract_decay_characteristics(audio_norm)
        mfcc_features = self._extract_mfcc_features(audio_norm)

        # Combine into single feature object
        return AdvancedAudioFeatures(
            # Temporal
            duration_ms=temporal_features['duration_ms'],
            attack_time_ms=temporal_features['attack_time_ms'],
            decay_time_ms=temporal_features['decay_time_ms'],
            kurtosis=temporal_features['kurtosis'],

            # Spectral
            spectral_centroid=spectral_features['centroid'],
            spectral_bandwidth=spectral_features['bandwidth'],
            spectral_rolloff=spectral_features['rolloff'],
            spectral_flatness=spectral_features['flatness'],
            spectral_flux=spectral_features['flux'],
            spectral_centroid_variance=spectral_features['centroid_variance'],

            # Energy
            rms_peak=energy_features['rms_peak'],
            rms_mean=energy_features['rms_mean'],
            peak_to_mean_ratio=energy_features['peak_to_mean_ratio'],
            onset_strength=energy_features['onset_strength'],

            # Harmonic
            harmonic_to_noise_ratio=harmonic_features['hnr'],
            has_pitch=harmonic_features['has_pitch'],
            zero_crossing_rate=harmonic_features['zcr'],

            # Frequency distribution
            low_freq_ratio=freq_dist['low'],
            mid_freq_ratio=freq_dist['mid'],
            high_freq_ratio=freq_dist['high'],

            # Decay
            has_ringing=decay_features['has_ringing'],
            decay_type=decay_features['decay_type'],

            # MFCCs
            mfcc_1=mfcc_features[0],
            mfcc_2=mfcc_features[1],
            mfcc_3=mfcc_features[2],
            mfcc_4=mfcc_features[3],
            mfcc_5=mfcc_features[4],
        )

    def features_to_vector(self, features: AdvancedAudioFeatures) -> np.ndarray:
        """Convert features to numpy array for ML."""
        return np.array([
            features.duration_ms,
            features.attack_time_ms,
            features.decay_time_ms,
            features.kurtosis,
            features.spectral_centroid,
            features.spectral_bandwidth,
            features.spectral_rolloff,
            features.spectral_flatness,
            features.spectral_flux,
            features.spectral_centroid_variance,
            features.rms_peak,
            features.rms_mean,
            features.peak_to_mean_ratio,
            features.onset_strength,
            features.harmonic_to_noise_ratio,
            float(features.has_pitch),
            features.zero_crossing_rate,
            features.low_freq_ratio,
            features.mid_freq_ratio,
            features.high_freq_ratio,
            float(features.has_ringing),
            features.mfcc_1,
            features.mfcc_2,
            features.mfcc_3,
            features.mfcc_4,
            features.mfcc_5,
        ])

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]."""
        max_val = np.max(np.abs(audio))
        return audio / (max_val + 1e-8) if max_val > 0 else audio

    def _extract_temporal_features(self, audio: np.ndarray) -> Dict:
        """Extract temporal characteristics."""
        duration_ms = len(audio) / self.sample_rate * 1000

        # Compute envelope
        envelope = np.abs(librosa.stft(audio, n_fft=512))
        envelope_mean = np.mean(envelope, axis=0)

        # Find peak
        peak_idx = np.argmax(envelope_mean)
        peak_value = envelope_mean[peak_idx]

        # Attack time (time to reach 90% of peak)
        attack_threshold = peak_value * 0.9
        attack_portion = envelope_mean[:peak_idx+1]
        attack_indices = np.where(attack_portion < attack_threshold)[0]
        attack_time_ms = (len(attack_indices) / len(envelope_mean) * duration_ms) if len(attack_indices) > 0 else 0

        # Decay time (time from peak to 10% of peak)
        decay_threshold = peak_value * 0.1
        decay_portion = envelope_mean[peak_idx:]
        decay_indices = np.where(decay_portion > decay_threshold)[0]
        decay_time_ms = (len(decay_indices) / len(envelope_mean) * duration_ms) if len(decay_indices) > 0 else 0

        # Kurtosis (measure of spikiness)
        kurtosis_value = float(stats.kurtosis(audio))

        return {
            'duration_ms': duration_ms,
            'attack_time_ms': attack_time_ms,
            'decay_time_ms': decay_time_ms,
            'kurtosis': kurtosis_value,
        }

    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Extract spectral characteristics."""
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        centroid_mean = float(np.mean(centroid))
        centroid_variance = float(np.std(centroid) / (centroid_mean + 1e-8))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        bandwidth_mean = float(np.mean(bandwidth))

        # Spectral rolloff (frequency containing 85% of energy)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, roll_percent=0.85)[0]
        rolloff_mean = float(np.mean(rolloff))

        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        flatness_mean = float(np.mean(flatness))

        # Spectral flux (stability)
        stft = librosa.stft(audio, n_fft=512, hop_length=128)
        magnitude = np.abs(stft)
        flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0) if magnitude.shape[1] > 1 else np.array([0])
        flux_mean = float(np.mean(flux))

        return {
            'centroid': centroid_mean,
            'bandwidth': bandwidth_mean,
            'rolloff': rolloff_mean,
            'flatness': flatness_mean,
            'flux': flux_mean,
            'centroid_variance': centroid_variance,
        }

    def _extract_energy_features(self, audio: np.ndarray) -> Dict:
        """Extract energy characteristics."""
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        rms_peak = float(np.max(rms))
        rms_mean = float(np.mean(rms))
        peak_to_mean_ratio = rms_peak / (rms_mean + 1e-8)

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        onset_strength = float(np.max(onset_env))

        return {
            'rms_peak': rms_peak,
            'rms_mean': rms_mean,
            'peak_to_mean_ratio': peak_to_mean_ratio,
            'onset_strength': onset_strength,
        }

    def _extract_harmonic_features(self, audio: np.ndarray) -> Dict:
        """Extract harmonic/pitch characteristics."""
        # Harmonic-to-noise ratio (simplified)
        autocorr = librosa.autocorrelate(audio)
        if len(autocorr) > 1:
            peak = np.max(autocorr[1:len(autocorr)//2])
            noise_floor = np.median(autocorr[1:])
            hnr = float(peak / (noise_floor + 1e-8))
        else:
            hnr = 0.0

        # Pitch detection
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            max_magnitude = float(np.max(magnitudes)) if magnitudes.size > 0 else 0.0
            has_pitch = max_magnitude > 0.1
        except:
            has_pitch = False

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = float(np.mean(zcr))

        return {
            'hnr': hnr,
            'has_pitch': has_pitch,
            'zcr': zcr_mean,
        }

    def _extract_frequency_distribution(self, audio: np.ndarray) -> Dict:
        """Analyze frequency distribution."""
        # Compute power spectral density
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=min(512, len(audio)))

        total_energy = np.sum(psd) + 1e-8

        # Low frequency (< 1000 Hz)
        low_mask = freqs < 1000
        low_energy = np.sum(psd[low_mask])
        low_ratio = float(low_energy / total_energy)

        # Mid frequency (1000-3000 Hz)
        mid_mask = (freqs >= 1000) & (freqs < 3000)
        mid_energy = np.sum(psd[mid_mask])
        mid_ratio = float(mid_energy / total_energy)

        # High frequency (>= 3000 Hz)
        high_mask = freqs >= 3000
        high_energy = np.sum(psd[high_mask])
        high_ratio = float(high_energy / total_energy)

        return {
            'low': low_ratio,
            'mid': mid_ratio,
            'high': high_ratio,
        }

    def _extract_decay_characteristics(self, audio: np.ndarray) -> Dict:
        """Analyze decay behavior (detect metallic ringing)."""
        # Compute envelope
        envelope = np.abs(librosa.stft(audio, n_fft=512))
        envelope_mean = np.mean(envelope, axis=0)

        if len(envelope_mean) < 10:
            return {
                'has_ringing': False,
                'decay_type': 'unknown',
            }

        # Find peak
        peak_idx = np.argmax(envelope_mean)

        # Analyze decay portion
        decay_portion = envelope_mean[peak_idx:]

        if len(decay_portion) > 5:
            # Check for ringing (periodic oscillations)
            decay_autocorr = np.correlate(decay_portion, decay_portion, mode='same')
            decay_autocorr = decay_autocorr / (decay_autocorr[len(decay_autocorr)//2] + 1e-8)

            # Count peaks above threshold
            peaks_above_threshold = np.sum(decay_autocorr > 0.3)
            has_ringing = peaks_above_threshold > 5

            # Classify decay type
            decay_length = len(decay_portion)
            total_length = len(envelope_mean)
            decay_ratio = decay_length / total_length

            if decay_ratio < 0.3:
                decay_type = 'fast'
            elif has_ringing:
                decay_type = 'ringing'
            else:
                decay_type = 'slow'
        else:
            has_ringing = False
            decay_type = 'unknown'

        return {
            'has_ringing': has_ringing,
            'decay_type': decay_type,
        }

    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCCs for timbre characterization."""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=5)
            mfcc_mean = np.mean(mfccs, axis=1)
            return mfcc_mean
        except:
            return np.zeros(5)


def demonstrate_features():
    """Demonstrate feature extraction on sample audio."""
    import sounddevice as sd

    print("Recording 2 seconds of audio...")
    print("Make a tongue click sound!")

    sample_rate = 44100
    duration = 2.0

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype='float32')
    sd.wait()

    print("\nExtracting features...")
    extractor = AdvancedFeatureExtractor(sample_rate=sample_rate)

    # Analyze the audio
    audio_flat = audio.flatten()
    features = extractor.extract_all_features(audio_flat)

    print("\n" + "="*70)
    print("EXTRACTED FEATURES")
    print("="*70)
    print(f"\nTemporal:")
    print(f"  Duration: {features.duration_ms:.1f} ms")
    print(f"  Attack time: {features.attack_time_ms:.1f} ms")
    print(f"  Decay time: {features.decay_time_ms:.1f} ms")
    print(f"  Kurtosis: {features.kurtosis:.2f}")

    print(f"\nSpectral:")
    print(f"  Centroid: {features.spectral_centroid:.0f} Hz")
    print(f"  Bandwidth: {features.spectral_bandwidth:.0f} Hz")
    print(f"  Rolloff: {features.spectral_rolloff:.0f} Hz")
    print(f"  Flatness: {features.spectral_flatness:.3f}")
    print(f"  Flux: {features.spectral_flux:.3f}")

    print(f"\nEnergy:")
    print(f"  Peak/Mean ratio: {features.peak_to_mean_ratio:.2f}")
    print(f"  Onset strength: {features.onset_strength:.2f}")

    print(f"\nHarmonic:")
    print(f"  HNR: {features.harmonic_to_noise_ratio:.2f}")
    print(f"  Has pitch: {features.has_pitch}")

    print(f"\nFrequency Distribution:")
    print(f"  Low (<1kHz): {features.low_freq_ratio*100:.1f}%")
    print(f"  Mid (1-3kHz): {features.mid_freq_ratio*100:.1f}%")
    print(f"  High (>3kHz): {features.high_freq_ratio*100:.1f}%")

    print(f"\nDecay:")
    print(f"  Has ringing: {features.has_ringing}")
    print(f"  Decay type: {features.decay_type}")

    print("\n" + "="*70)


if __name__ == "__main__":
    demonstrate_features()
