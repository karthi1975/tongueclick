#!/usr/bin/env python3
"""
Tongue Click Sample Collector

Records short clips of your tongue clicks for training.
Each clip is ~3 seconds. Make ONE click per clip.

Usage:
    # Collect 50 tongue click samples (guided, with countdown)
    python collect_clicks.py

    # Collect 100 samples, 2 seconds each
    python collect_clicks.py --count 100 --duration 2

    # Review what you've collected
    python collect_clicks.py --review
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
import os
import time
from datetime import datetime


def collect_clicks(output_dir='training_data/positives', count=50,
                   duration=3.0, sample_rate=44100, device=None):
    """Record tongue click samples one at a time with countdown."""
    os.makedirs(output_dir, exist_ok=True)

    # Count existing samples
    existing = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])

    print(f"\n{'='*70}")
    print(f"TONGUE CLICK SAMPLE COLLECTOR")
    print(f"{'='*70}")
    print(f"Output       : {output_dir}/")
    print(f"Existing     : {existing} samples")
    print(f"To collect   : {count} new samples")
    print(f"Duration     : {duration}s per clip")
    print(f"{'='*70}")
    print(f"\nInstructions:")
    print(f"  - Make ONE clear tongue click per recording")
    print(f"  - Wait for 'RECORDING' before clicking")
    print(f"  - After playback, press Enter to KEEP or 's' to SKIP")
    print(f"  - Press Ctrl+C to stop anytime")
    print(f"{'='*70}\n")

    input("Press Enter to start...")

    saved = 0
    skipped = 0

    try:
        for i in range(count):
            print(f"\n--- Sample {i+1}/{count} (saved: {saved}, skipped: {skipped}) ---")

            # Countdown
            for sec in [3, 2, 1]:
                print(f"  {sec}...")
                time.sleep(1)

            print(f"  >> RECORDING ({duration}s) -- CLICK NOW!")

            # Record
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=device
            )
            sd.wait()

            peak = np.max(np.abs(audio))
            print(f"  Recording done. (peak energy: {peak:.3f})")

            if peak < 0.002:
                print(f"  Too quiet -- skipping.")
                skipped += 1
                continue

            # Playback
            print(f"  Playing back...")
            sd.play(audio, sample_rate)
            sd.wait()

            # Confirm
            response = input(f"  Keep this sample? [Enter=YES / s=skip]: ").strip().lower()

            if response == 's':
                print(f"  Skipped.")
                skipped += 1
            else:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filepath = os.path.join(output_dir, f"click_{timestamp_str}.wav")
                sf.write(filepath, audio, sample_rate)
                saved += 1
                print(f"  Saved! ({saved} total)")

    except KeyboardInterrupt:
        print(f"\n\nStopped early.")

    total = existing + saved
    print(f"\n{'='*70}")
    print(f"COLLECTION DONE")
    print(f"{'='*70}")
    print(f"Saved this session : {saved}")
    print(f"Skipped            : {skipped}")
    print(f"Total in folder    : {total}")
    print(f"{'='*70}")

    if total < 30:
        print(f"\nYou have {total} samples. Aim for 50+ for good training results.")
        print(f"Run again to collect more: python collect_clicks.py --count {50 - total}")
    else:
        print(f"\nYou have enough samples to train!")
        print(f"Next step:")
        print(f"  python retrain_model.py \\")
        print(f"    --positive {output_dir} \\")
        print(f"    --negative training_data/auto_collected")

    print()


def review(output_dir='training_data/positives'):
    """Show what's been collected."""
    print(f"\n{'='*70}")
    print(f"TONGUE CLICK SAMPLES")
    print(f"{'='*70}\n")

    if not os.path.exists(output_dir):
        print(f"  No samples at {output_dir}/")
        return

    wav_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    print(f"  Total samples : {len(wav_files)}")

    if wav_files:
        total_bytes = sum(
            os.path.getsize(os.path.join(output_dir, f)) for f in wav_files
        )
        print(f"  Disk usage    : {total_bytes / (1024*1024):.1f} MB")
        print(f"  First         : {wav_files[0]}")
        print(f"  Last          : {wav_files[-1]}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect tongue click samples for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 50 samples (default)
  python collect_clicks.py

  # Collect 100 samples, 2 seconds each
  python collect_clicks.py --count 100 --duration 2

  # Review
  python collect_clicks.py --review

Full workflow:
  1. python auto_collect.py --hours 24    # negatives (overnight)
  2. python collect_clicks.py             # positives (10 min)
  3. python retrain_model.py --positive training_data/positives --negative training_data/auto_collected
        """
    )

    parser.add_argument('--count', type=int, default=50,
                        help='Number of samples to collect (default: 50)')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Seconds per clip (default: 3)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Sample rate (default: 44100)')
    parser.add_argument('--output-dir', type=str,
                        default='training_data/positives',
                        help='Output directory (default: training_data/positives)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--review', action='store_true',
                        help='Review collected samples')

    args = parser.parse_args()

    if args.review:
        review(args.output_dir)
    else:
        collect_clicks(
            output_dir=args.output_dir,
            count=args.count,
            duration=args.duration,
            sample_rate=args.sample_rate,
            device=args.device,
        )


if __name__ == "__main__":
    main()
