#!/usr/bin/env python
"""
Test microphone to see if audio is being captured
"""

import numpy as np
import sounddevice as sd
import time

print("=" * 70)
print("MICROPHONE TEST - Shows raw audio levels")
print("=" * 70)
print()

# List devices
print("Available audio devices:")
print(sd.query_devices())
print("=" * 70)
print()

print("Testing microphone for 10 seconds...")
print("Make sounds: speak, click, type, etc.")
print()

sample_rate = 44100
chunk_duration = 0.1
chunk_samples = int(sample_rate * chunk_duration)

chunk_count = 0
energy_levels = []

def audio_callback(indata, frames, time_info, status):
    global chunk_count
    chunk_count += 1

    if status:
        print(f"Status: {status}")

    audio = indata[:, 0]

    # Calculate basic metrics
    max_val = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    energy_levels.append(max_val)

    # Show every chunk
    if chunk_count % 10 == 0:  # Every second
        print(f"[{chunk_count//10}s] Max: {max_val:.4f}, RMS: {rms:.4f}", end="")

        # Visual bar
        bar_length = int(max_val * 50)
        print(f" {'█' * bar_length}")

try:
    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=sample_rate,
                       blocksize=chunk_samples):
        sd.sleep(10 * 1000)
except KeyboardInterrupt:
    print("\nStopped by user")

print()
print("=" * 70)
print(f"Total chunks processed: {chunk_count}")
print(f"Max energy seen: {max(energy_levels) if energy_levels else 0:.4f}")
print(f"Average energy: {np.mean(energy_levels) if energy_levels else 0:.4f}")
print()

if max(energy_levels) < 0.01:
    print("⚠️  WARNING: Very low audio levels detected!")
    print("    Check:")
    print("    1. Microphone permissions (System Preferences > Security)")
    print("    2. Microphone volume/gain")
    print("    3. Is correct input device selected?")
elif max(energy_levels) < 0.1:
    print("⚠️  Audio detected but quiet")
    print("    Try speaking louder or moving closer to microphone")
else:
    print("✓ Microphone is working!")
    print(f"  Peak audio level: {max(energy_levels):.4f}")

print("=" * 70)
