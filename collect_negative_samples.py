#!/usr/bin/env python3
"""
Negative Sample Collection Tool
Collects and labels non-tongue-click sounds for training
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
import os
from datetime import datetime
import json


class NegativeSampleCollector:
    """Collect negative examples for tongue click detector training."""

    CATEGORIES = [
        'dog_bark',
        'utensils_metal',
        'sink_water',
        'sink_cleaning',
        'bed_creak',
        'keyboard_typing',
        'door_close',
        'ambient_noise',
        'human_speech',
        'other',
    ]

    def __init__(self, output_dir='training_data/negatives', sample_rate=44100):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.metadata = []

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        for category in self.CATEGORIES:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    def record_sample(self, category: str, duration: float = 15.0,
                     description: str = ""):
        """Record a negative sample."""
        if category not in self.CATEGORIES:
            print(f"Invalid category. Choose from: {self.CATEGORIES}")
            return

        print(f"\n{'='*70}")
        print(f"Recording: {category}")
        print(f"Duration: {duration} seconds")
        if description:
            print(f"Description: {description}")
        print(f"{'='*70}\n")

        input("Press Enter to start recording...")

        print("🔴 RECORDING... Make the sound now!")

        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        print("✓ Recording complete!")

        # Play back
        print("\nPlayback (verify quality)...")
        sd.play(audio, self.sample_rate)
        sd.wait()

        # Confirm save
        response = input("\nSave this sample? (y/n): ").strip().lower()

        if response == 'y':
            self._save_sample(audio, category, description)
            print("✓ Sample saved!")
        else:
            print("✗ Sample discarded")

    def _save_sample(self, audio: np.ndarray, category: str, description: str):
        """Save audio sample and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{category}_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, category, filename)

        # Save audio file
        sf.write(filepath, audio, self.sample_rate)

        # Save metadata
        metadata_entry = {
            'filename': filename,
            'category': category,
            'description': description,
            'timestamp': timestamp,
            'duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'label': 0,  # 0 = negative (not a tongue click)
        }
        self.metadata.append(metadata_entry)

        # Update metadata file
        metadata_file = os.path.join(self.output_dir, 'metadata.json')

        # Load existing metadata if present
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = []

        existing_metadata.append(metadata_entry)

        with open(metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)

        # Clear in-memory list since data is persisted to disk
        # Prevents memory growth during long collection sessions
        self.metadata.clear()

    def batch_collect(self, category: str, num_samples: int = 10,
                     duration: float = 3.0):
        """Collect multiple samples of the same category."""
        print(f"\n{'='*70}")
        print(f"BATCH COLLECTION MODE")
        print(f"Category: {category}")
        print(f"Number of samples: {num_samples}")
        print(f"Duration per sample: {duration} seconds")
        print(f"{'='*70}\n")

        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            self.record_sample(category, duration)

            if i < num_samples - 1:
                print("\nPrepare for next sample...")

    def show_statistics(self):
        """Show collection statistics."""
        metadata_file = os.path.join(self.output_dir, 'metadata.json')

        if not os.path.exists(metadata_file):
            print("No samples collected yet.")
            return

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"\n{'='*70}")
        print("COLLECTION STATISTICS")
        print(f"{'='*70}\n")

        # Count by category
        category_counts = {}
        total_duration = 0

        for entry in metadata:
            category = entry['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            total_duration += entry['duration']

        print(f"Total samples: {len(metadata)}")
        print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"\nBreakdown by category:")

        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            print(f"  {category:20s}: {count:3d} samples")

        print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect negative samples for tongue click detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 15 seconds of dog barking
  python collect_negative_samples.py --category dog_bark --duration 15

  # Batch collect 10 utensil sounds (3 seconds each)
  python collect_negative_samples.py --category utensils_metal --batch 10 --duration 3

  # Show statistics
  python collect_negative_samples.py --stats

  # Interactive mode
  python collect_negative_samples.py --interactive

Available categories:
  - dog_bark: Dog barking sounds
  - utensils_metal: Plates, spoons, forks clanking
  - sink_water: Water running, dripping
  - sink_cleaning: Scrubbing, washing dishes
  - bed_creak: Bed or furniture creaking
  - keyboard_typing: Keyboard and mouse clicks
  - door_close: Doors closing, handles
  - ambient_noise: Background noise
  - human_speech: Speaking, coughing, etc.
  - other: Other sounds
        """
    )

    parser.add_argument('--category', choices=NegativeSampleCollector.CATEGORIES,
                       help='Sound category to record')
    parser.add_argument('--duration', type=float, default=15.0,
                       help='Recording duration in seconds (default: 15)')
    parser.add_argument('--batch', type=int,
                       help='Batch mode: collect N samples')
    parser.add_argument('--description', type=str, default='',
                       help='Description of the sound')
    parser.add_argument('--output-dir', type=str, default='training_data/negatives',
                       help='Output directory for samples')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    collector = NegativeSampleCollector(output_dir=args.output_dir)

    if args.stats:
        collector.show_statistics()
        return

    if args.interactive:
        # Interactive mode
        while True:
            print(f"\n{'='*70}")
            print("INTERACTIVE NEGATIVE SAMPLE COLLECTION")
            print(f"{'='*70}")
            print("\nAvailable categories:")
            for i, cat in enumerate(collector.CATEGORIES, 1):
                print(f"  {i}. {cat}")
            print(f"  {len(collector.CATEGORIES)+1}. Show statistics")
            print(f"  {len(collector.CATEGORIES)+2}. Exit")

            try:
                choice = input("\nSelect category (number): ").strip()
                choice_num = int(choice)

                if choice_num == len(collector.CATEGORIES) + 1:
                    collector.show_statistics()
                    continue
                elif choice_num == len(collector.CATEGORIES) + 2:
                    print("\nGoodbye!")
                    break
                elif 1 <= choice_num <= len(collector.CATEGORIES):
                    category = collector.CATEGORIES[choice_num - 1]
                    duration = float(input("Duration (seconds, default 15): ") or "15")
                    description = input("Description (optional): ").strip()

                    collector.record_sample(category, duration, description)
                else:
                    print("Invalid choice!")
            except (ValueError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

    elif args.category:
        if args.batch:
            collector.batch_collect(args.category, args.batch, args.duration)
        else:
            collector.record_sample(args.category, args.duration, args.description)

        collector.show_statistics()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
