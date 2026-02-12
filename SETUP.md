# Tongue Click Detector - Setup Instructions

## Prerequisites

- **Python 3.11 or 3.12** (Required - Python 3.13 is NOT supported due to llvmlite compatibility issues)
- macOS, Linux, or Windows
- Audio input device (microphone)

## Quick Start

### 1. Check Python Version

First, verify you have Python 3.11 or 3.12 installed:

```bash
python3.11 --version
# or
python3.12 --version
```

If you don't have Python 3.11/3.12, install it:

**macOS (using Homebrew):**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and select Python 3.11.x

### 2. Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd /path/to/tongue_click
python3.11 -m venv venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- scipy (scientific computing)
- sounddevice (audio I/O)
- librosa (audio feature extraction)
- numba (performance optimization)
- audioread (audio file reading)
- scikit-learn (machine learning utilities)
- pyautogui (mouse automation)

### 5. Run the Detector

**Real-time detection:**
```bash
python tongue_click_detector.py
```

**File analysis:**
```bash
# Edit tongue_click_detector.py and uncomment the file analysis example
# Or use the API programmatically
```

## Usage Examples

### Basic Real-time Detection

```python
from tongue_click_detector import DetectorFactory, DetectorConfig

# Create configuration
config = DetectorConfig(
    sample_rate=44100,
    threshold=0.3,
    confidence_threshold=0.6
)

# Create detector
detector = DetectorFactory.create_default_detector(config)

# Create real-time audio source (10 seconds)
audio_source = DetectorFactory.create_real_time_source(config, duration=10)

# Run detection
results = audio_source.process_audio(detector=detector)

print(f"Detected {len(results)} clicks")
```

### Analyze Audio File

```python
from tongue_click_detector import DetectorFactory, DetectorConfig

# Create configuration
config = DetectorConfig()

# Create detector
detector = DetectorFactory.create_default_detector(config)

# Create file source
file_source = DetectorFactory.create_file_source("audio_file.wav", config)

# Analyze file
results = file_source.process_audio(detector=detector)
```

### Custom Event Handler

```python
from tongue_click_detector import (
    DetectorFactory,
    DetectorConfig,
    IEventHandler,
    ClickDetectionResult
)

class CustomHandler(IEventHandler):
    def on_click_detected(self, result: ClickDetectionResult) -> None:
        print(f"Custom action! Confidence: {result.confidence_score}")
        # Add your custom logic here

# Use the handler
handler = CustomHandler()
audio_source = DetectorFactory.create_real_time_source(config, duration=10)
results = audio_source.process_audio(
    detector=detector,
    callback=lambda r: handler.on_click_detected(r)
)
```

## Configuration Options

The `DetectorConfig` class allows you to customize detection parameters:

```python
config = DetectorConfig(
    sample_rate=44100,              # Audio sample rate (Hz)
    threshold=0.3,                  # Onset detection threshold
    onset_weight=0.4,               # Weight for onset strength
    frequency_weight=0.3,           # Weight for frequency content
    impulsive_weight=0.3,           # Weight for impulsive nature
    confidence_threshold=0.6,       # Minimum confidence for detection
    min_energy_threshold=0.01,      # Minimum audio energy
    chunk_duration=0.1              # Audio chunk size (seconds)
)
```

### Tuning for Your Environment

- **Increase `threshold`** (0.3 → 0.5) if detecting too many false positives
- **Decrease `threshold`** (0.3 → 0.2) if missing real clicks
- **Adjust `confidence_threshold`** to control detection sensitivity
- **Modify `chunk_duration`** for different time resolutions

## Troubleshooting

### Issue: `OSError: Could not find/load shared object file`

**Solution:** You're using Python 3.13, which is not supported. Use Python 3.11 or 3.12:

```bash
# Remove existing venv
rm -rf venv

# Create new venv with Python 3.11
python3.11 -m venv venv

# Activate and reinstall
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: No audio input detected

**Solution:** Check available audio devices:

```python
import sounddevice as sd
print(sd.query_devices())
```

Select the correct input device in your code:

```python
import sounddevice as sd
sd.default.device = 0  # Replace 0 with your device index
```

### Issue: ImportError for librosa or numba

**Solution:** Ensure all dependencies are installed:

```bash
pip install --upgrade -r requirements.txt
```

### Issue: PortAudio errors on Linux

**Solution:** Install PortAudio library:

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Issue: Permission denied on macOS for microphone

**Solution:** Grant microphone permissions in System Preferences:
1. System Preferences → Security & Privacy → Privacy → Microphone
2. Enable access for Terminal or your IDE

## Deactivating Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Upgrading Dependencies

To update all dependencies to their latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

## Cleaning Up

To remove the virtual environment:

```bash
# Deactivate first
deactivate

# Remove the venv directory
rm -rf venv  # macOS/Linux
# or
rmdir /s venv  # Windows
```

## Development

### Running Tests

```bash
# Add your test commands here when tests are created
python -m pytest tests/
```

### Code Style

This project follows SOLID principles:
- **Single Responsibility Principle**: Each class has one clear purpose
- **Open/Closed Principle**: Easy to extend without modifying existing code
- **Liskov Substitution Principle**: Abstract base classes ensure proper substitution
- **Interface Segregation Principle**: Focused interfaces for specific needs
- **Dependency Inversion Principle**: Depends on abstractions, uses dependency injection

## Additional Resources

- [librosa documentation](https://librosa.org/doc/latest/index.html)
- [sounddevice documentation](https://python-sounddevice.readthedocs.io/)
- [Python 3.11 documentation](https://docs.python.org/3.11/)

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the code documentation in `tongue_click_detector.py`
3. Check GitHub issues (if applicable)

## License

[Add your license information here]
