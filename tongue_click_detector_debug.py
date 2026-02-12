#!/usr/bin/env python
"""
Debug version of tongue click detector - shows all audio events
"""

from tongue_click_detector import *

if __name__ == "__main__":
    print("=" * 70)
    print("TONGUE CLICK DETECTOR - DEBUG MODE")
    print("This will show ALL audio events (including near-misses)")
    print("=" * 70)
    print()

    # Start with LENIENT settings to see everything
    config = DetectorConfig(
        sample_rate=44100,
        threshold=0.3,                    # Low - see more events
        confidence_threshold=0.5,         # Low - see more events
        min_spectral_centroid=2000,       # Low - see more events
        min_peak_to_mean_ratio=3.0,       # Low - see more events
        min_energy_threshold=0.01         # Low - see quiet sounds
    )

    detector = DetectorFactory.create_default_detector(config)

    # List devices
    print("Available audio devices:")
    print(sd.query_devices())
    print("=" * 70)
    print()

    # Create custom callback that shows details
    class DebugHandler:
        def __init__(self):
            self.event_count = 0

        def on_event(self, result):
            self.event_count += 1
            f = result.features

            status = "✓ CLICK" if result.is_click else "  EVENT"
            print(f"{status} #{self.event_count}:")
            print(f"  Onset strength: {f.onset_strength:.3f} (need > {config.threshold})")
            print(f"  Frequency:      {f.spectral_centroid:.0f} Hz (need > {config.min_spectral_centroid})")
            print(f"  Peak/Mean:      {f.peak_to_mean_ratio:.2f}x (need > {config.min_peak_to_mean_ratio})")
            print(f"  Confidence:     {result.confidence_score:.3f} (need > {config.confidence_threshold})")
            print(f"  RMS Peak:       {f.rms_energy_peak:.4f}")
            print(f"  RMS Mean:       {f.rms_energy_mean:.4f}")
            print()

    handler = DebugHandler()

    print("Listening for 10 seconds...")
    print("Try: tongue clicks, speaking, keyboard typing, etc.")
    print()

    # Modified audio source to show all events
    start_time = time.time()
    results = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}")

        audio_chunk = indata[:, 0]

        if detector.has_sufficient_energy(audio_chunk):
            result = detector.detect(audio_chunk)

            # Show all events with some confidence
            if result.confidence_score > 0.3 or result.is_click:
                result.timestamp = time.time() - start_time
                handler.on_event(result)
                if result.is_click:
                    results.append(result)

    chunk_samples = int(config.sample_rate * config.chunk_duration)

    try:
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=config.sample_rate,
                          blocksize=chunk_samples):
            sd.sleep(10 * 1000)
    except KeyboardInterrupt:
        print("\nStopped by user")

    print("=" * 70)
    print(f"Total events detected: {handler.event_count}")
    print(f"Clicks detected: {len(results)}")
    print("=" * 70)
