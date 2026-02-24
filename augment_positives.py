#!/usr/bin/env python3
"""
Audio Augmentation for Tongue Click Training Data

Takes existing positive samples and generates augmented versions using:
- Pitch shifting
- Time stretching
- Volume variation (including low-volume simulation)
- Background noise addition
- Combined augmentations
"""

import numpy as np
import soundfile as sf
import librosa
import os
import argparse
from datetime import datetime


def pitch_shift(audio, sr, n_steps):
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(audio, rate):
    """Stretch time by rate factor (>1 = faster, <1 = slower)."""
    return librosa.effects.time_stretch(y=audio, rate=rate)


def volume_scale(audio, factor):
    """Scale volume by factor."""
    return audio * factor


def add_noise(audio, noise_level):
    """Add Gaussian noise."""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise.astype(audio.dtype)


def augment_file(audio, sr, rng):
    """Generate multiple augmented versions of a single audio file.

    Returns list of (augmented_audio, augmentation_label) tuples.
    """
    augmented = []

    # 1. Pitch shifts: -3 to +3 semitones
    for steps in [-3, -2, -1, 1, 2, 3]:
        aug = pitch_shift(audio, sr, steps)
        augmented.append((aug, f'pitch_{steps:+d}'))

    # 2. Time stretches: 0.8x to 1.2x speed
    for rate in [0.85, 0.9, 1.1, 1.15]:
        aug = time_stretch(audio, rate)
        augmented.append((aug, f'stretch_{rate}'))

    # 3. Volume variations
    # Low volume simulation (like the reference file)
    for factor in [0.1, 0.15, 0.2, 0.3]:
        aug = volume_scale(audio, factor)
        augmented.append((aug, f'vol_{factor}'))

    # Louder
    for factor in [1.3, 1.5, 1.8]:
        aug = volume_scale(audio, factor)
        aug = np.clip(aug, -1.0, 1.0)
        augmented.append((aug, f'vol_{factor}'))

    # 4. Noise addition
    for noise_level in [0.002, 0.005, 0.01]:
        aug = add_noise(audio, noise_level)
        augmented.append((aug, f'noise_{noise_level}'))

    # 5. Combined augmentations (pitch + volume)
    for steps in [-2, 2]:
        for factor in [0.2, 0.5, 1.5]:
            aug = pitch_shift(audio, sr, steps)
            aug = volume_scale(aug, factor)
            aug = np.clip(aug, -1.0, 1.0)
            augmented.append((aug, f'pitch_{steps:+d}_vol_{factor}'))

    # 6. Combined (stretch + noise)
    for rate in [0.9, 1.1]:
        for noise_level in [0.003, 0.008]:
            aug = time_stretch(audio, rate)
            aug = add_noise(aug, noise_level)
            augmented.append((aug, f'stretch_{rate}_noise_{noise_level}'))

    # 7. Random combinations for more variety
    for i in range(25):
        steps = rng.uniform(-2.5, 2.5)
        factor = rng.uniform(0.15, 1.8)
        noise_level = rng.uniform(0.001, 0.008)

        aug = pitch_shift(audio, sr, steps)
        aug = volume_scale(aug, factor)
        aug = add_noise(aug, noise_level)
        aug = np.clip(aug, -1.0, 1.0)
        augmented.append((aug, f'rand_{i}'))

    return augmented


def main():
    parser = argparse.ArgumentParser(
        description='Augment positive tongue click samples')
    parser.add_argument('--input-dir', default='training_data/positives',
                        help='Directory with original positive samples')
    parser.add_argument('--output-dir', default='training_data/positives',
                        help='Output directory (default: same as input)')
    parser.add_argument('--target', type=int, default=2000,
                        help='Target number of total samples (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load original files
    originals = []
    for f in sorted(os.listdir(input_dir)):
        if not f.endswith('.wav'):
            continue
        # Skip previously augmented files
        if '_aug_' in f:
            continue
        filepath = os.path.join(input_dir, f)
        audio, sr = sf.read(filepath)
        audio = audio.flatten()
        originals.append((f, audio, sr))

    print(f"Original samples: {len(originals)}")
    print(f"Target total:     {args.target}")

    # Count existing augmented files
    existing_aug = len([f for f in os.listdir(output_dir)
                        if f.endswith('.wav') and '_aug_' in f])
    current_total = len(originals) + existing_aug
    print(f"Existing augmented: {existing_aug}")
    print(f"Current total:    {current_total}")

    if current_total >= args.target:
        print(f"Already have {current_total} samples. Nothing to do.")
        return

    needed = args.target - len(originals)
    per_file = needed // len(originals) + 1

    print(f"Need ~{per_file} augmented versions per original file")
    print(f"\nAugmenting...")

    saved = 0
    for orig_name, audio, sr in originals:
        base_name = orig_name.replace('.wav', '')
        augmentations = augment_file(audio, sr, rng)

        for aug_audio, label in augmentations:
            if saved + len(originals) >= args.target:
                break

            timestamp = datetime.now().strftime("%H%M%S_%f")
            out_name = f"{base_name}_aug_{label}_{timestamp}.wav"
            out_path = os.path.join(output_dir, out_name)

            sf.write(out_path, aug_audio, sr)
            saved += 1

        if saved + len(originals) >= args.target:
            break

    total = len(originals) + saved
    print(f"\nDone!")
    print(f"  New augmented files: {saved}")
    print(f"  Total samples:      {total}")


if __name__ == '__main__':
    main()
