# Quick Start Guide

Get up and running with the Tongue Click Detector in under 5 minutes!

## Step 1: Set Up Virtual Environment (Recommended)

### Option A: Quick Setup (macOS/Linux)

```bash
./activate.sh
```

This will:
- Activate the virtual environment
- Display available commands
- Keep your system Python clean

### Option B: Manual Setup

**Create and activate virtual environment:**

macOS/Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

**Note**: A virtual environment (`venv/`) has already been created with all dependencies installed! If you need to reinstall:

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- scipy (signal processing)
- sounddevice (audio I/O)
- librosa (audio analysis)
- And other required packages

## Step 2: Verify Installation

Run the test script to ensure everything is working:

```bash
python test_basic.py
```

You should see:
```
✓ ALL TESTS PASSED
The tongue click detector is ready to use!
```

## Step 3: Try It Out!

### Option A: Interactive Demo (Recommended for first-time users)

```bash
python demo.py
```

This launches an interactive menu where you can:
- Test real-time detection
- Analyze audio files
- View audio devices

### Option B: Quick Real-Time Test

```bash
python demo.py --mode realtime --duration 10
```

This will:
1. Listen to your microphone for 10 seconds
2. Detect tongue clicks in real-time
3. Display confidence scores

**Tips for best results:**
- Position microphone 6-12 inches from your mouth
- Make clear, sharp tongue click sounds
- Minimize background noise

## Common Issues

### Issue: "No module named 'numpy'"
**Solution**: Run `pip install -r requirements.txt`

### Issue: "No audio devices found"
**Solution**:
- macOS: Grant microphone permissions in System Preferences
- Linux: Add your user to audio group: `sudo usermod -a -G audio $USER`

### Issue: Too many false detections
**Solution**: Increase threshold
```bash
python demo.py --mode realtime --threshold 0.4
```

### Issue: Missing real clicks
**Solution**: Decrease threshold
```bash
python demo.py --mode realtime --threshold 0.2
```

## What's Next?

- Read [README.md](README.md) for detailed documentation
- Check [requirements.md](requirements.md) for technical specifications
- Integrate into your own project (see examples below)

## Integration Example

```python
from tongue_click_detector import TongueClickDetector

# Create detector
detector = TongueClickDetector(threshold=0.3)

# Define what happens when a click is detected
def on_click_detected(timestamp, confidence):
    print(f"Click! Time: {timestamp:.2f}s, Confidence: {confidence:.2f}")
    # Add your custom action here!

# Start listening
detector.real_time_detection(duration=30, callback=on_click_detected)
```

## Need Help?

1. Check the [README.md](README.md) troubleshooting section
2. Run `python demo.py --mode devices` to list audio devices
3. Verify Python version: `python --version` (needs 3.8+)

Happy clicking! 👅🔊
