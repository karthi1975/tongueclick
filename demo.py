#!/usr/bin/env python3
"""
Demo script for the Tongue Click Detector

This script provides interactive examples of using the TongueClickDetector
for both real-time detection and audio file analysis.
"""

import argparse
import sys
from tongue_click_detector import TongueClickDetector, list_audio_devices


def demo_realtime(duration=10, threshold=0.3, sample_rate=44100):
    """
    Demonstrate real-time tongue click detection.

    Args:
        duration (int): How long to listen in seconds
        threshold (float): Detection sensitivity (0-1)
        sample_rate (int): Audio sampling rate
    """
    print("\n" + "=" * 70)
    print("REAL-TIME TONGUE CLICK DETECTION DEMO")
    print("=" * 70)

    detector = TongueClickDetector(sample_rate=sample_rate, threshold=threshold)

    # Custom callback to track statistics
    click_count = [0]

    def on_click_detected(timestamp, score):
        click_count[0] += 1

    print(f"\nConfiguration:")
    print(f"  Duration: {duration} seconds")
    print(f"  Threshold: {threshold}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print("\nInstructions:")
    print("  - Position your microphone close to your mouth")
    print("  - Make clear tongue click sounds")
    print("  - Press Ctrl+C to stop early\n")

    input("Press Enter to start listening...")

    clicks = detector.real_time_detection(duration=duration, callback=on_click_detected)

    print("\n" + "=" * 70)
    print(f"Session complete! Detected {len(clicks)} tongue clicks")
    print("=" * 70)


def demo_file_analysis(filepath, threshold=0.3, sample_rate=44100):
    """
    Demonstrate audio file analysis.

    Args:
        filepath (str): Path to audio file
        threshold (float): Detection sensitivity (0-1)
        sample_rate (int): Audio sampling rate
    """
    print("\n" + "=" * 70)
    print("AUDIO FILE ANALYSIS DEMO")
    print("=" * 70)

    detector = TongueClickDetector(sample_rate=sample_rate, threshold=threshold)

    print(f"\nConfiguration:")
    print(f"  File: {filepath}")
    print(f"  Threshold: {threshold}")
    print(f"  Sample Rate: {sample_rate} Hz\n")

    clicks = detector.analyze_audio_file(filepath)

    if clicks:
        print("\n" + "=" * 70)
        print(f"Analysis complete! Found {len(clicks)} clicks")
        print("=" * 70)

        if len(clicks) > 0:
            avg_confidence = sum(score for _, score in clicks) / len(clicks)
            print(f"\nStatistics:")
            print(f"  Average confidence: {avg_confidence:.2f}")
            print(f"  Timestamps: {[f'{t:.2f}s' for t, _ in clicks[:5]]}")
            if len(clicks) > 5:
                print(f"  ... and {len(clicks) - 5} more")


def show_devices():
    """Display available audio devices."""
    print("\n" + "=" * 70)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 70 + "\n")
    list_audio_devices()
    print("\n" + "=" * 70)


def interactive_menu():
    """Display an interactive menu for demo options."""
    print("\n" + "=" * 70)
    print("TONGUE CLICK DETECTOR - INTERACTIVE DEMO")
    print("=" * 70)
    print("\nSelect an option:")
    print("  1. Real-time detection (10 seconds)")
    print("  2. Real-time detection (30 seconds)")
    print("  3. Analyze audio file")
    print("  4. Show available audio devices")
    print("  5. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == "1":
                demo_realtime(duration=10)
            elif choice == "2":
                demo_realtime(duration=30)
            elif choice == "3":
                filepath = input("Enter audio file path: ").strip()
                demo_file_analysis(filepath)
            elif choice == "4":
                show_devices()
            elif choice == "5":
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Tongue Click Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python demo.py

  # Real-time detection for 15 seconds
  python demo.py --mode realtime --duration 15

  # Analyze an audio file
  python demo.py --mode file --input recording.wav

  # Show available audio devices
  python demo.py --mode devices

  # Adjust sensitivity
  python demo.py --mode realtime --threshold 0.2
        """
    )

    parser.add_argument(
        '--mode',
        choices=['realtime', 'file', 'devices', 'interactive'],
        default='interactive',
        help='Operation mode (default: interactive)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Duration for real-time detection in seconds (default: 10)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input audio file path for file mode'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Detection threshold 0-1 (default: 0.3, lower = more sensitive)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Audio sample rate in Hz (default: 44100)'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'interactive':
            interactive_menu()
        elif args.mode == 'realtime':
            demo_realtime(args.duration, args.threshold, args.sample_rate)
        elif args.mode == 'file':
            if not args.input:
                print("Error: --input required for file mode")
                sys.exit(1)
            demo_file_analysis(args.input, args.threshold, args.sample_rate)
        elif args.mode == 'devices':
            show_devices()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
