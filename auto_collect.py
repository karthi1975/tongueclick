#!/usr/bin/env python3
"""
Auto Negative Sample Collector - Unattended 24h+ collection.

Listens to the microphone continuously and saves any sound above
the energy threshold as a negative (non-click) .wav sample.
No model needed. No detection. Just records household sounds.

Run overnight, then retrain your model with the new negatives.

Usage:
    # Collect negatives for 24 hours (default)
    python auto_collect.py

    # Collect for 8 hours
    python auto_collect.py --hours 8

    # More sensitive (captures quieter sounds)
    python auto_collect.py --min-energy 0.01

    # Review what was collected
    python auto_collect.py --review
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
import os
import gc
import json
import time
from datetime import datetime, timedelta


def run_collection(duration, output_dir, sample_rate=44100,
                   min_energy=0.02, chunk_duration=0.5):
    """
    Record all sounds above energy threshold to disk.
    Memory-safe for 24h+ -- no audio kept in memory.
    """
    os.makedirs(output_dir, exist_ok=True)

    chunk_samples = int(sample_rate * chunk_duration)
    start_time = time.time()
    end_time = start_time + duration
    saved_count = [0]
    skipped_count = [0]
    gc_counter = [0]
    last_save_time = [0.0]
    min_save_interval = chunk_duration * 0.8  # Avoid overlapping clips

    hours = duration / 3600
    print(f"\n{'='*70}")
    print(f"NEGATIVE SAMPLE COLLECTOR (unattended)")
    print(f"{'='*70}")
    print(f"Duration     : {hours:.1f} hours")
    print(f"Output       : {output_dir}/")
    print(f"Min energy   : {min_energy}")
    print(f"Chunk length : {chunk_duration}s")
    print(f"Sample rate  : {sample_rate} Hz")
    print(f"Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ends at      : {(datetime.now() + timedelta(seconds=duration)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPress Ctrl+C to stop early.")
    print(f"{'='*70}\n")

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  Stream: {status}")

        current_time = time.time()
        audio_chunk = indata[:, 0]

        # Skip quiet audio
        if np.max(np.abs(audio_chunk)) < min_energy:
            skipped_count[0] += 1
            return

        # Rate-limit saves
        if current_time - last_save_time[0] < min_save_interval:
            return
        last_save_time[0] = current_time

        # Save to disk immediately -- nothing kept in memory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(output_dir, f"neg_{timestamp_str}.wav")
        sf.write(filepath, audio_chunk, sample_rate)

        saved_count[0] += 1
        elapsed = current_time - start_time
        print(f"  [{timedelta(seconds=int(elapsed))}] "
              f"Saved #{saved_count[0]}: {filepath}")

        # Periodic GC every ~60s
        gc_counter[0] += 1
        if gc_counter[0] >= int(60 / chunk_duration):
            gc_counter[0] = 0
            gc.collect()

    try:
        with sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=sample_rate,
                            blocksize=chunk_samples):
            while time.time() < end_time:
                remaining = end_time - time.time()
                sleep_ms = min(10000, int(remaining * 1000))
                if sleep_ms <= 0:
                    break
                sd.sleep(sleep_ms)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Ran for          : {timedelta(seconds=int(elapsed))}")
    print(f"Samples saved    : {saved_count[0]}")
    print(f"Chunks skipped   : {skipped_count[0]} (below energy threshold)")
    print(f"Output directory : {output_dir}/")
    print(f"{'='*70}")

    # Save stats
    stats = {
        'saved': saved_count[0],
        'skipped': skipped_count[0],
        'elapsed_seconds': elapsed,
        'started': datetime.fromtimestamp(start_time).isoformat(),
        'ended': datetime.now().isoformat(),
        'sample_rate': sample_rate,
        'chunk_duration': chunk_duration,
        'min_energy': min_energy,
    }
    stats_file = os.path.join(output_dir, 'collection_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nNext steps:")
    print(f"  1. Listen to samples and delete any tongue clicks that snuck in")
    print(f"  2. Retrain the model:")
    print(f"     python retrain_model.py \\")
    print(f"       --positive training_data/positives \\")
    print(f"       --negative {output_dir}")
    print()


def review_collected(output_dir):
    """Show summary of collected samples."""
    print(f"\n{'='*70}")
    print(f"COLLECTED SAMPLES")
    print(f"{'='*70}\n")

    if not os.path.exists(output_dir):
        print(f"  No samples found at {output_dir}/")
        return

    wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    print(f"  Total .wav files: {len(wav_files)}")

    if wav_files:
        # Estimate disk usage
        total_bytes = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in wav_files
        )
        print(f"  Disk usage      : {total_bytes / (1024*1024):.1f} MB")

        # Time range
        wav_files.sort()
        print(f"  First sample    : {wav_files[0]}")
        print(f"  Last sample     : {wav_files[-1]}")

    stats_file = os.path.join(output_dir, 'collection_stats.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n  Last run: {stats.get('started', '?')} to {stats.get('ended', '?')}")
        print(f"  Duration: {timedelta(seconds=int(stats.get('elapsed_seconds', 0)))}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect negative (non-click) audio samples unattended for 24h+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect for 24 hours (default)
  python auto_collect.py

  # Collect for 8 hours
  python auto_collect.py --hours 8

  # More sensitive (captures quieter sounds like keyboard)
  python auto_collect.py --min-energy 0.01

  # Longer clips (1 second each instead of 0.5s)
  python auto_collect.py --chunk-duration 1.0

  # Review what was collected
  python auto_collect.py --review

After collection:
  1. Listen to a few samples, delete any actual tongue clicks
  2. Retrain: python retrain_model.py --positive training_data/positives --negative training_data/auto_collected
        """
    )

    parser.add_argument('--hours', type=float, default=24,
                        help='Collection duration in hours (default: 24)')
    parser.add_argument('--min-energy', type=float, default=0.02,
                        help='Min energy to save a clip (default: 0.02, lower=more sensitive)')
    parser.add_argument('--chunk-duration', type=float, default=0.5,
                        help='Audio chunk length in seconds (default: 0.5)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate (default: 44100)')
    parser.add_argument('--output-dir', type=str,
                        default='training_data/auto_collected',
                        help='Output directory (default: training_data/auto_collected)')
    parser.add_argument('--review', action='store_true',
                        help='Review collected samples instead of collecting')

    args = parser.parse_args()

    if args.review:
        review_collected(args.output_dir)
        return

    duration_seconds = int(args.hours * 3600)
    run_collection(
        duration=duration_seconds,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_energy=args.min_energy,
        chunk_duration=args.chunk_duration,
    )


if __name__ == "__main__":
    main()
