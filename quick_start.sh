#!/bin/bash
# Quick Start Script for Tongue Click Detector Training
# This script guides you through the complete workflow

set -e  # Exit on error

echo "======================================================================"
echo "TONGUE CLICK DETECTOR - QUICK START GUIDE"
echo "======================================================================"
echo ""
echo "This script will help you:"
echo "  1. Collect negative training samples"
echo "  2. Train an ML model"
echo "  3. Test the detector"
echo ""
echo "Prerequisites:"
echo "  - Python 3.7+"
echo "  - Microphone"
echo "  - Existing tongue click samples in 'training_data/positives/'"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if training_data/positives exists
if [ ! -d "training_data/positives" ]; then
    echo "WARNING: training_data/positives/ directory not found."
    echo "Creating directory..."
    mkdir -p training_data/positives
    echo ""
    echo "Please add your tongue click .wav files to:"
    echo "  training_data/positives/"
    echo ""
    read -p "Press Enter when you've added your files..."
fi

# Count positive samples
POSITIVE_COUNT=$(find training_data/positives -name "*.wav" -o -name "*.WAV" | wc -l)
echo ""
echo "Found $POSITIVE_COUNT tongue click samples"

if [ "$POSITIVE_COUNT" -lt 5 ]; then
    echo ""
    echo "WARNING: Very few positive samples found!"
    echo "For best results, you should have at least 20-50 tongue click recordings."
    echo ""
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 0
    fi
fi

echo ""
echo "======================================================================"
echo "STEP 1: COLLECT NEGATIVE SAMPLES"
echo "======================================================================"
echo ""
echo "You need to collect sounds that are NOT tongue clicks but might"
echo "trigger false positives. Examples:"
echo "  - Dog barking"
echo "  - Utensils clanking"
echo "  - Water/sink sounds"
echo "  - Bed creaking"
echo "  - Keyboard typing"
echo ""
echo "Target: 100-200 negative samples (10-20 per category)"
echo ""

# Check if negatives already exist
if [ -d "training_data/negatives" ]; then
    NEGATIVE_COUNT=$(find training_data/negatives -name "*.wav" -o -name "*.WAV" | wc -l)
    echo "Found $NEGATIVE_COUNT existing negative samples"

    if [ "$NEGATIVE_COUNT" -gt 50 ]; then
        read -p "Skip negative sample collection? (y/n): " SKIP_NEG
        if [ "$SKIP_NEG" == "y" ]; then
            echo "Skipping negative sample collection..."
            goto_training=1
        fi
    fi
fi

if [ -z "$goto_training" ]; then
    echo ""
    read -p "Start collecting negative samples? (y/n): " COLLECT

    if [ "$COLLECT" == "y" ]; then
        echo ""
        echo "Launching interactive collector..."
        echo ""
        python3 collect_negative_samples.py --interactive
    else
        echo "Skipping negative sample collection."
        echo "You can run it later with:"
        echo "  python3 collect_negative_samples.py --interactive"
    fi
fi

# Check if we have enough data
NEGATIVE_COUNT=$(find training_data/negatives -name "*.wav" -o -name "*.WAV" 2>/dev/null | wc -l || echo 0)

echo ""
echo "======================================================================"
echo "DATA SUMMARY"
echo "======================================================================"
echo "Positive samples: $POSITIVE_COUNT"
echo "Negative samples: $NEGATIVE_COUNT"
echo ""

if [ "$NEGATIVE_COUNT" -lt 20 ]; then
    echo "WARNING: You have very few negative samples!"
    echo "The model may not generalize well."
    echo ""
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 0
    fi
fi

echo ""
echo "======================================================================"
echo "STEP 2: TRAIN THE MODEL"
echo "======================================================================"
echo ""
read -p "Start model training? (y/n): " TRAIN

if [ "$TRAIN" == "y" ]; then
    echo ""
    echo "Training Random Forest model..."
    echo ""

    python3 retrain_model.py \
        --positive training_data/positives \
        --negative training_data/negatives \
        --model random_forest \
        --output-dir models

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Model training complete!"
    else
        echo ""
        echo "✗ Model training failed!"
        exit 1
    fi
else
    echo "Skipping model training."
    echo "You can train later with:"
    echo "  python3 retrain_model.py --positive training_data/positives --negative training_data/negatives"
    exit 0
fi

echo ""
echo "======================================================================"
echo "STEP 3: TEST THE DETECTOR"
echo "======================================================================"
echo ""
echo "The model is now trained and ready to use!"
echo ""
echo "Test options:"
echo "  1. Real-time detection (30 seconds)"
echo "  2. Analyze audio file"
echo "  3. Skip testing for now"
echo ""
read -p "Select option (1-3): " TEST_OPTION

case $TEST_OPTION in
    1)
        echo ""
        echo "Starting real-time detection..."
        echo "Try making tongue clicks AND the sounds that previously caused false positives."
        echo ""
        read -p "Press Enter to start..."
        python3 ml_detector.py --mode realtime --duration 30
        ;;
    2)
        echo ""
        read -p "Enter path to audio file: " AUDIO_FILE
        python3 ml_detector.py --mode file --input "$AUDIO_FILE"
        ;;
    3)
        echo "Skipping testing."
        ;;
esac

echo ""
echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Your tongue click detector is ready to use!"
echo ""
echo "Usage examples:"
echo ""
echo "  # Real-time detection"
echo "  python3 ml_detector.py --mode realtime --duration 60"
echo ""
echo "  # Analyze file"
echo "  python3 ml_detector.py --mode file --input recording.wav"
echo ""
echo "  # Adjust sensitivity (0.5=sensitive, 0.9=strict)"
echo "  python3 ml_detector.py --mode realtime --threshold 0.75"
echo ""
echo "If you still get false positives:"
echo "  1. Collect more negative examples of those sounds"
echo "  2. Retrain the model"
echo "  3. Increase the confidence threshold (--threshold 0.85)"
echo ""
echo "See TRAINING_WORKFLOW.md for detailed documentation."
echo ""
echo "Good luck! 🎯"
