#!/usr/bin/env python3
"""
Analyze saved click audio files to identify false positives.

Compares feature profiles of detected clicks to help distinguish
real tongue clicks from false triggers (podcast, TV, etc.)

Usage:
    # Analyze all saved clicks
    python analyze_clicks.py

    # Analyze specific files
    python analyze_clicks.py detected_clicks/click_0185*.wav detected_clicks/click_0186*.wav

    # Analyze a range by click number
    python analyze_clicks.py --range 185 188

    # Compare against known good clicks (from training data)
    python analyze_clicks.py --range 185 188 --compare-good training_data/positives_16k/

    # Flag likely false positives (outliers)
    python analyze_clicks.py --flag-outliers

    # Move flagged false positives to hard negatives folder
    python analyze_clicks.py --range 185 188 --move-to-negatives
"""

import argparse
import glob
import os
import shutil
import sys
import numpy as np
import librosa
from pathlib import Path
from advanced_features import AdvancedFeatureExtractor


def analyze_file(filepath, extractor, sample_rate=44100):
    """Analyze a single audio file and return features + metadata."""
    audio, sr = librosa.load(filepath, sr=sample_rate)
    duration_ms = len(audio) / sample_rate * 1000

    features = extractor.extract_all_features(audio)
    vector = extractor.features_to_vector(features)

    return {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'duration_ms': duration_ms,
        'features': features,
        'vector': vector,
        'audio': audio,
        'rms': float(np.sqrt(np.mean(audio ** 2))),
    }


def print_analysis(result, label=""):
    """Print detailed analysis of a click."""
    f = result['features']
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}{result['filename']}")
    print(f"  Duration: {result['duration_ms']:.0f}ms | RMS: {result['rms']:.4f}")
    print(f"  Attack: {f.attack_time_ms:.1f}ms | Decay: {f.decay_time_ms:.1f}ms | Kurtosis: {f.kurtosis:.1f}")
    print(f"  Spectral centroid: {f.spectral_centroid:.0f}Hz | Bandwidth: {f.spectral_bandwidth:.0f}Hz")
    print(f"  Flatness: {f.spectral_flatness:.4f} | Flux: {f.spectral_flux:.2f}")
    print(f"  Freq: low={f.low_freq_ratio:.1%} mid={f.mid_freq_ratio:.1%} high={f.high_freq_ratio:.1%}")
    print(f"  Has pitch: {f.has_pitch} | ZCR: {f.zero_crossing_rate:.3f} | HNR: {f.harmonic_to_noise_ratio:.2f}")
    print(f"  MFCCs: [{f.mfcc_1:.1f}, {f.mfcc_2:.1f}, {f.mfcc_3:.1f}, {f.mfcc_4:.1f}, {f.mfcc_5:.1f}]")
    print(f"  Ringing: {f.has_ringing} | Decay type: {f.decay_type}")


def find_clicks_by_range(click_dir, start, end):
    """Find click files by number range."""
    files = []
    for i in range(start, end + 1):
        pattern = os.path.join(click_dir, f"click_{i:04d}_*.wav")
        matches = glob.glob(pattern)
        files.extend(matches)
    return sorted(files)


def compute_stats(results):
    """Compute mean and std of feature vectors."""
    vectors = np.array([r['vector'] for r in results])
    return np.mean(vectors, axis=0), np.std(vectors, axis=0)


def flag_outliers(all_results, threshold=2.0):
    """Flag clicks that are outliers compared to the group."""
    if len(all_results) < 5:
        print("  Need at least 5 clicks to detect outliers.")
        return []

    vectors = np.array([r['vector'] for r in all_results])
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0) + 1e-8

    flagged = []
    for r in all_results:
        z_scores = np.abs((r['vector'] - mean) / std)
        max_z = np.max(z_scores)
        mean_z = np.mean(z_scores)
        if mean_z > threshold:
            flagged.append((r, mean_z, max_z))

    return sorted(flagged, key=lambda x: -x[1])


FEATURE_NAMES = [
    'duration_ms', 'attack_time_ms', 'decay_time_ms', 'kurtosis',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'spectral_flatness', 'spectral_flux', 'centroid_variance',
    'rms_peak', 'rms_mean', 'peak_to_mean_ratio', 'onset_strength',
    'hnr', 'has_pitch', 'zcr',
    'low_freq_ratio', 'mid_freq_ratio', 'high_freq_ratio',
    'has_ringing', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
]


def compare_to_reference(target_results, ref_results):
    """Compare target clicks against reference good clicks."""
    ref_mean, ref_std = compute_stats(ref_results)

    print("\n" + "=" * 70)
    print("COMPARISON TO REFERENCE (good tongue clicks)")
    print("=" * 70)

    for r in target_results:
        z_scores = np.abs((r['vector'] - ref_mean) / (ref_std + 1e-8))
        mean_z = np.mean(z_scores)
        top_deviations = np.argsort(z_scores)[::-1][:5]

        verdict = "LIKELY GOOD" if mean_z < 1.5 else "SUSPICIOUS" if mean_z < 2.5 else "LIKELY FALSE POSITIVE"
        print(f"\n  {r['filename']} -> {verdict} (avg z-score: {mean_z:.2f})")
        print(f"  Top deviating features:")
        for idx in top_deviations:
            print(f"    {FEATURE_NAMES[idx]:25s}: {r['vector'][idx]:8.2f} "
                  f"(ref: {ref_mean[idx]:.2f} +/- {ref_std[idx]:.2f}, "
                  f"z={z_scores[idx]:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze saved click audio files for false positives",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('files', nargs='*', help='Specific WAV files to analyze')
    parser.add_argument('--range', nargs=2, type=int, metavar=('START', 'END'),
                        help='Analyze clicks by number range (e.g., --range 185 188)')
    parser.add_argument('--click-dir', default='detected_clicks',
                        help='Directory with saved clicks (default: detected_clicks)')
    parser.add_argument('--compare-good', type=str,
                        help='Directory with known good tongue clicks for comparison')
    parser.add_argument('--flag-outliers', action='store_true',
                        help='Flag likely false positives among all saved clicks')
    parser.add_argument('--outlier-threshold', type=float, default=2.0,
                        help='Z-score threshold for outlier detection (default: 2.0)')
    parser.add_argument('--move-to-negatives', action='store_true',
                        help='Move analyzed files to training_data/hard_negatives/')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Sample rate for feature extraction (default: 44100)')

    args = parser.parse_args()

    extractor = AdvancedFeatureExtractor(sample_rate=args.sample_rate)

    # Collect files to analyze
    files = []
    if args.files:
        files = args.files
    elif args.range:
        files = find_clicks_by_range(args.click_dir, args.range[0], args.range[1])
        if not files:
            print(f"No clicks found in range {args.range[0]}-{args.range[1]}")
            return
    elif args.flag_outliers:
        files = sorted(glob.glob(os.path.join(args.click_dir, "click_*.wav")))
        if not files:
            print(f"No click files found in {args.click_dir}/")
            return
    else:
        files = sorted(glob.glob(os.path.join(args.click_dir, "click_*.wav")))
        if not files:
            print(f"No click files found in {args.click_dir}/")
            return

    print(f"\nAnalyzing {len(files)} click(s)...")

    # Analyze each file
    results = []
    for filepath in files:
        try:
            result = analyze_file(filepath, extractor, args.sample_rate)
            results.append(result)
            print_analysis(result)
        except Exception as e:
            print(f"\n  ERROR analyzing {filepath}: {e}")

    if not results:
        print("No files could be analyzed.")
        return

    # Compare to reference good clicks
    if args.compare_good:
        print(f"\nLoading reference clicks from {args.compare_good}...")
        ref_files = sorted(glob.glob(os.path.join(args.compare_good, "*.wav")))[:50]
        ref_results = []
        for rf in ref_files:
            try:
                ref_results.append(analyze_file(rf, extractor, args.sample_rate))
            except Exception:
                continue
        if ref_results:
            compare_to_reference(results, ref_results)
        else:
            print("  No reference files could be loaded.")

    # Flag outliers
    if args.flag_outliers:
        print("\n" + "=" * 70)
        print("OUTLIER DETECTION")
        print("=" * 70)
        flagged = flag_outliers(results, args.outlier_threshold)
        if flagged:
            print(f"\n  Found {len(flagged)} potential false positive(s):")
            for r, mean_z, max_z in flagged:
                print(f"    {r['filename']} (avg z={mean_z:.2f}, max z={max_z:.2f})")
        else:
            print("\n  No obvious outliers detected.")

    # Move to hard negatives
    if args.move_to_negatives:
        neg_dir = Path("training_data/hard_negatives")
        neg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMoving {len(files)} file(s) to {neg_dir}/")
        for filepath in files:
            dest = neg_dir / os.path.basename(filepath)
            shutil.move(filepath, dest)
            print(f"  Moved: {os.path.basename(filepath)}")
        print("Done. Use these as hard negatives when retraining.")

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        vectors = np.array([r['vector'] for r in results])
        print(f"  Files analyzed: {len(results)}")
        print(f"  Avg spectral centroid: {np.mean(vectors[:, 4]):.0f} Hz")
        print(f"  Avg flatness: {np.mean(vectors[:, 7]):.4f}")
        print(f"  Has pitch: {sum(1 for r in results if r['features'].has_pitch)}/{len(results)}")
        print(f"  Avg kurtosis: {np.mean(vectors[:, 3]):.1f}")
        print(f"  Avg ZCR: {np.mean(vectors[:, 16]):.3f}")


if __name__ == '__main__':
    main()
