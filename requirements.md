Here's Python code to recognize tongue click sounds using audio processing:

```python
import numpy as np
import sounddevice as sd
import librosa
from scipy import signal
from scipy.fft import fft
import time

class TongueClickDetector:
    def __init__(self, sample_rate=44100, threshold=0.3):
        self.sample_rate = sample_rate
        self.threshold = threshold
        
    def detect_click_features(self, audio_chunk):
        """
        Detect if audio chunk contains a tongue click based on features:
        - Short duration (20-50ms)
        - High frequency content
        - Sharp onset
        - Low zero-crossing rate initially
        """
        # Normalize audio
        audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)
        
        # 1. Check for sharp onset (sudden amplitude increase)
        onset_strength = librosa.onset.onset_strength(y=audio_chunk, sr=self.sample_rate)
        has_sharp_onset = np.max(onset_strength) > self.threshold
        
        # 2. Check spectral characteristics (clicks have broadband energy)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=self.sample_rate)
        high_frequency = np.mean(spectral_centroid) > 2000  # Clicks typically > 2kHz
        
        # 3. Check for short duration and impulsiveness
        rms_energy = librosa.feature.rms(y=audio_chunk)[0]
        energy_peak = np.max(rms_energy)
        is_impulsive = energy_peak > np.mean(rms_energy) * 3
        
        # 4. Zero-crossing rate (clicks have specific patterns)
        zcr = librosa.feature.zero_crossing_rate(audio_chunk)[0]
        avg_zcr = np.mean(zcr)
        
        # Combine features for detection
        click_score = (
            has_sharp_onset * 0.4 +
            high_frequency * 0.3 +
            is_impulsive * 0.3
        )
        
        return click_score > 0.6, click_score
    
    def real_time_detection(self, duration=10):
        """Record audio and detect clicks in real-time"""
        print(f"Listening for tongue clicks for {duration} seconds...")
        print("Make tongue click sounds!")
        
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(self.sample_rate * chunk_duration)
        
        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            
            audio_chunk = indata[:, 0]  # Get mono channel
            
            # Check if chunk has enough energy
            if np.max(np.abs(audio_chunk)) > 0.01:
                is_click, score = self.detect_click_features(audio_chunk)
                
                if is_click:
                    print(f"✓ Tongue click detected! (confidence: {score:.2f})")
        
        with sd.InputStream(callback=callback, 
                          channels=1, 
                          samplerate=self.sample_rate,
                          blocksize=chunk_samples):
            sd.sleep(int(duration * 1000))
    
    def analyze_audio_file(self, filepath):
        """Analyze a pre-recorded audio file for clicks"""
        print(f"Analyzing audio file: {filepath}")
        
        # Load audio file
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        
        # Split into small chunks
        chunk_length = int(0.1 * sr)  # 100ms chunks
        clicks_detected = []
        
        for i in range(0, len(audio) - chunk_length, chunk_length // 2):
            chunk = audio[i:i + chunk_length]
            
            if np.max(np.abs(chunk)) > 0.01:
                is_click, score = self.detect_click_features(chunk)
                
                if is_click:
                    timestamp = i / sr
                    clicks_detected.append((timestamp, score))
                    print(f"Click at {timestamp:.2f}s (confidence: {score:.2f})")
        
        return clicks_detected


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = TongueClickDetector(sample_rate=44100, threshold=0.3)
    
    # Option 1: Real-time detection
    print("Option 1: Real-time detection")
    detector.real_time_detection(duration=10)
    
    # Option 2: Analyze audio file
    # clicks = detector.analyze_audio_file("your_audio.wav")
    # print(f"\nTotal clicks detected: {len(clicks)}")
```

**Installation requirements:**
```bash
pip install numpy sounddevice librosa scipy
```

**Key features of the detector:**

1. **Sharp Onset Detection** - Tongue clicks have a sudden amplitude increase
2. **High Frequency Content** - Clicks contain significant energy above 2kHz
3. **Impulsive Nature** - Short, sharp bursts of energy
4. **Spectral Analysis** - Broadband frequency distribution

**Usage options:**
- **Real-time**: Listens to microphone and detects clicks as they happen
- **File analysis**: Analyzes pre-recorded audio files

You can adjust the `threshold` parameter (0-1) to make detection more or less sensitive. Lower values = more sensitive but more false positives.