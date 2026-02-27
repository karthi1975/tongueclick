#!/usr/bin/env python3
"""
Hard Negative Sample Collector

Records specific sounds that cause false positives (e.g., nose breathing)
and saves them as negative training samples in auto_collected format.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
import os
import time
from datetime import datetime


def collect(label, output_dir='training_data/auto_collected',
            duration=60, chunk_duration=0.5, sample_rate=44100,
            min_energy=0.005, device=None):
    """Record and save hard negative samples in chunks."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"HARD NEGATIVE COLLECTOR: {label}")
    print(f"{'='*70}")
    print(f"Duration     : {duration}s")
    print(f"Chunk size   : {chunk_duration}s")
    print(f"Output       : {output_dir}/")
    print(f"Min energy   : {min_energy}")
    print(f"{'='*70}")
    print(f"\nMake '{label}' sounds continuously for {duration} seconds.")
    print(f"Press Ctrl+C to stop early.\n")

    input("Press Enter to start recording...")

    chunk_samples = int(chunk_duration * sample_rate)
    saved = 0
    skipped = 0

    print(f"RECORDING... make '{label}' sounds now!\n")

    try:
        # Record full duration
        total_samples = int(duration * sample_rate)
        audio = sd.rec(total_samples, samplerate=sample_rate,
                       channels=1, dtype='float32', device=device)
        sd.wait()

        print("Recording done. Processing chunks...\n")

        # Split into chunks and save those with enough energy
        for i in range(0, len(audio) - chunk_samples, chunk_samples):
            chunk = audio[i:i + chunk_samples].flatten()
            energy = np.max(np.abs(chunk))

            if energy >= min_energy:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"neg_{label}_{timestamp}_{saved}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, chunk, sample_rate)
                saved += 1
            else:
                skipped += 1

    except KeyboardInterrupt:
        print("\nStopped early.")

    print(f"\n{'='*70}")
    print(f"COLLECTION DONE")
    print(f"{'='*70}")
    print(f"Saved:   {saved} chunks")
    print(f"Skipped: {skipped} (below energy threshold)")
    print(f"{'='*70}\n")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description='Collect hard negative samples for retraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 60s of nose breathing
  python collect_hard_negatives.py --label nose_breathing --duration 60

  # Record 30s of coughing
  python collect_hard_negatives.py --label coughing --duration 30

  # Record with lower energy threshold (capture quieter sounds)
  python collect_hard_negatives.py --label nose_breathing --duration 60 --min-energy 0.002
        """)

    parser.add_argument('--label', type=str, required=True,
                        help='Label for the sound (e.g., nose_breathing)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Recording duration in seconds (default: 60)')
    parser.add_argument('--chunk-duration', type=float, default=0.5,
                        help='Chunk size in seconds (default: 0.5)')
    parser.add_argument('--output-dir', type=str,
                        default='training_data/auto_collected',
                        help='Output directory')
    parser.add_argument('--min-energy', type=float, default=0.005,
                        help='Minimum energy to save a chunk (default: 0.005)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Sample rate (default: 44100)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device index')

    args = parser.parse_args()

    collect(
        label=args.label,
        output_dir=args.output_dir,
        duration=args.duration,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        min_energy=args.min_energy,
        device=args.device,
    )

    print("Next steps:")
    print("  1. Optionally record more hard negatives")
    print("  2. Retrain the model:")
    print("     python retrain_model.py \\")
    print("       --positive training_data/positives \\")
    print("       --negative training_data/auto_collected")


if __name__ == '__main__':
    main()
