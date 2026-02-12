#!/usr/bin/env python
"""
Analyze tongue clicks - shows exact feature values
"""

from tongue_click_detector import *

print("=" * 70)
print("TONGUE CLICK ANALYZER - Shows feature values for ALL sounds")
print("=" * 70)
print()

# Very lenient config
config = DetectorConfig(
    sample_rate=44100,
    threshold=0.1,                    # Very low
    confidence_threshold=0.1,         # Very low
    min_spectral_centroid=1000,       # Very low
    min_peak_to_mean_ratio=1.5,       # Very low
    min_energy_threshold=0.005        # Very low
)

detector = DetectorFactory.create_default_detector(config)

print("Available audio devices:")
print(sd.query_devices())
print("=" * 70)
print()

print("Listening for 15 seconds...")
print("Make 3-4 LOUD tongue clicks, then speak, then type")
print()

start_time = time.time()
all_events = []

def audio_callback(indata, frames, time_info, status):
    audio_chunk = indata[:, 0]

    max_val = np.max(np.abs(audio_chunk))

    # Show EVERY chunk with reasonable energy
    if max_val > 0.01:  # Only if louder than background
        features = detector.feature_extractor.extract_features(audio_chunk)
        timestamp = time.time() - start_time

        all_events.append({
            'time': timestamp,
            'max': max_val,
            'features': features
        })

        print(f"[{timestamp:5.2f}s] "
              f"Max:{max_val:.3f} "
              f"Onset:{features.onset_strength:.2f} "
              f"Freq:{features.spectral_centroid:4.0f}Hz "
              f"Ratio:{features.peak_to_mean_ratio:.1f}x")

chunk_samples = int(config.sample_rate * config.chunk_duration)

try:
    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=config.sample_rate,
                       blocksize=chunk_samples):
        sd.sleep(15 * 1000)
except KeyboardInterrupt:
    print("\nStopped")

print()
print("=" * 70)
print(f"Total sounds detected: {len(all_events)}")

if all_events:
    # Find peak values
    max_onset = max(e['features'].onset_strength for e in all_events)
    max_freq = max(e['features'].spectral_centroid for e in all_events)
    max_ratio = max(e['features'].peak_to_mean_ratio for e in all_events)

    print()
    print("Peak values seen:")
    print(f"  Max onset strength: {max_onset:.2f}")
    print(f"  Max frequency:      {max_freq:.0f} Hz")
    print(f"  Max peak/mean ratio: {max_ratio:.1f}x")
    print()
    print("Recommended config for YOUR tongue clicks:")
    print(f"  threshold:              {max_onset * 0.7:.2f}")
    print(f"  min_spectral_centroid:  {max_freq * 0.8:.0f}")
    print(f"  min_peak_to_mean_ratio: {max_ratio * 0.8:.1f}")
    print(f"  confidence_threshold:   0.70")

print("=" * 70)
