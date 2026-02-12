# Tongue Click Sound Recognition

A Python-based system for detecting tongue click sounds in real-time audio streams or pre-recorded audio files using advanced audio signal processing techniques.

## Features

- **Real-time Detection**: Listen to microphone input and detect tongue clicks as they happen
- **File Analysis**: Analyze pre-recorded audio files for tongue click events
- **High Accuracy**: Uses multiple audio features for robust detection:
  - Sharp onset detection
  - High frequency content analysis
  - Impulsive energy patterns
  - Spectral characteristics
- **Configurable Sensitivity**: Adjust detection threshold to balance sensitivity and false positives
- **Interactive Demo**: User-friendly command-line interface for testing

## How It Works

Tongue clicks are characterized by unique acoustic properties:

1. **Short Duration**: Typically 20-50 milliseconds
2. **High Frequency Content**: Significant energy above 2kHz
3. **Sharp Onset**: Sudden amplitude increase
4. **Impulsive Nature**: Sharp bursts of energy with specific spectral distribution

The detector analyzes these features in real-time using:
- Librosa for audio feature extraction
- SciPy for signal processing
- NumPy for numerical computations

## Installation

### Prerequisites

- **Python 3.11 or 3.12** (Required - Python 3.13 is NOT supported due to llvmlite compatibility)
- pip package manager
- A working microphone (for real-time detection)

### Automated Setup (Recommended)

**macOS/Linux:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

The setup script will:
- Check for Python 3.11/3.12
- Create virtual environment
- Install all dependencies
- Verify installation

### Manual Setup

1. Clone or download this repository:
```bash
cd tongue_click
```

2. Create virtual environment with Python 3.11:
```bash
python3.11 -m venv venv
```

3. Activate virtual environment:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Platform-Specific Notes

**macOS**: You may need to grant microphone permissions to your terminal/Python application.

**Linux**: You might need to install PortAudio:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows**: Should work out of the box with pip installation.

## Usage

### Quick Start

Run the detector with default settings:
```bash
python tongue_click_detector.py
```

The detector will:
1. List available audio devices
2. Listen for 10 seconds
3. Detect and report tongue click sounds
4. Display confidence scores

### Using as a Python Module

The refactored code follows SOLID principles with dependency injection:

```python
from tongue_click_detector import DetectorFactory, DetectorConfig

# Create configuration
config = DetectorConfig(
    sample_rate=44100,
    threshold=0.3,
    confidence_threshold=0.6
)

# Create detector using factory
detector = DetectorFactory.create_default_detector(config)

# Real-time detection
audio_source = DetectorFactory.create_real_time_source(config, duration=10)
results = audio_source.process_audio(detector=detector)

print(f"Detected {len(results)} clicks")
```

### Analyze Audio File

```python
from tongue_click_detector import DetectorFactory, DetectorConfig

config = DetectorConfig()
detector = DetectorFactory.create_default_detector(config)

# Analyze file
file_source = DetectorFactory.create_file_source("audio.wav", config)
results = file_source.process_audio(detector=detector)
```

### Custom Event Handler

```python
from tongue_click_detector import IEventHandler, ClickDetectionResult

class MyHandler(IEventHandler):
    def on_click_detected(self, result: ClickDetectionResult) -> None:
        print(f"Custom action! Confidence: {result.confidence_score}")

# Use the handler
handler = MyHandler()
results = audio_source.process_audio(
    detector=detector,
    callback=lambda r: handler.on_click_detected(r)
)
```

## Configuration

The `DetectorConfig` class provides all configuration options:

```python
from tongue_click_detector import DetectorConfig

config = DetectorConfig(
    sample_rate=44100,              # Audio sampling rate (Hz)
    threshold=0.3,                  # Onset detection threshold
    onset_weight=0.4,               # Weight for onset strength
    frequency_weight=0.3,           # Weight for frequency content
    impulsive_weight=0.3,           # Weight for impulsive nature
    confidence_threshold=0.6,       # Minimum confidence for detection
    min_energy_threshold=0.01,      # Minimum audio energy
    chunk_duration=0.1              # Audio chunk size (seconds)
)
```

### Detection Parameters

- **sample_rate**: Audio sampling rate in Hz (default: 44100)
  - Higher rates provide better frequency resolution
  - 44100 Hz is standard CD quality and works well

- **threshold**: Onset detection sensitivity from 0 to 1 (default: 0.3)
  - Lower values = more sensitive (more detections, more false positives)
  - Higher values = less sensitive (fewer false positives, might miss clicks)
  - Recommended range: 0.2 - 0.4

- **confidence_threshold**: Minimum confidence score (default: 0.6)
  - Combined weighted score must exceed this to trigger detection
  - Range: 0.0 - 1.0

### Tuning Tips

1. **Too many false positives**:
   - Increase `threshold` (0.3 → 0.4)
   - Increase `confidence_threshold` (0.6 → 0.7)
2. **Missing clicks**:
   - Decrease `threshold` (0.3 → 0.2)
   - Decrease `confidence_threshold` (0.6 → 0.5)
3. **Noisy environment**: Increase threshold and ensure microphone is close
4. **Quiet clicks**: Decrease threshold and speak closer to microphone

## API Reference

### TongueClickDetector

Main class for tongue click detection.

#### `__init__(sample_rate=44100, threshold=0.3)`

Initialize the detector.

**Parameters:**
- `sample_rate` (int): Audio sampling rate in Hz
- `threshold` (float): Detection sensitivity (0-1)

#### `detect_click_features(audio_chunk)`

Analyze an audio chunk for tongue click features.

**Parameters:**
- `audio_chunk` (numpy.ndarray): Audio samples to analyze

**Returns:**
- `tuple`: (is_click: bool, confidence_score: float)

#### `real_time_detection(duration=10, callback=None)`

Record and analyze audio in real-time.

**Parameters:**
- `duration` (int): How long to listen in seconds
- `callback` (function): Optional callback(timestamp, score) for each detection

**Returns:**
- `list`: List of (timestamp, score) tuples for all detected clicks

#### `analyze_audio_file(filepath)`

Analyze a pre-recorded audio file.

**Parameters:**
- `filepath` (str): Path to audio file (WAV, MP3, etc.)

**Returns:**
- `list`: List of (timestamp, score) tuples for all detected clicks

### Helper Functions

#### `list_audio_devices()`

Display all available audio input/output devices.

## Architecture

The system is built with **SOLID principles** for maintainability and extensibility:

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to extend with new detectors/sources without modification
- **Liskov Substitution**: Abstract base classes ensure proper substitution
- **Interface Segregation**: Focused interfaces for specific needs
- **Dependency Inversion**: Depends on abstractions, uses dependency injection

### Key Components

- `IAudioFeatureExtractor`: Interface for feature extraction
- `LibrosaFeatureExtractor`: Extracts audio features using librosa
- `IClickClassifier`: Interface for click classification
- `WeightedFeatureClassifier`: Classifies using weighted features
- `TongueClickDetector`: Main detector orchestrating the pipeline
- `RealTimeAudioSource`: Real-time audio stream processing
- `AudioFileSource`: Pre-recorded audio file processing
- `DetectorFactory`: Simplified object creation with DI

## File Structure

```
tongue_click/
├── README.md                   # This file
├── SETUP.md                    # Detailed setup instructions
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script (macOS/Linux)
├── setup.bat                   # Automated setup script (Windows)
├── tongue_click_detector.py    # Main detector implementation
├── venv/                       # Virtual environment (Python 3.11)
└── venv_py313_backup/         # Backup of old venv (if exists)
```

## Troubleshooting

### OSError: Could not find/load shared object file

**Cause**: You're using Python 3.13, which is not supported by llvmlite yet.

**Solution**: Use Python 3.11 or 3.12:
```bash
# Remove existing venv
rm -rf venv

# Run automated setup with Python 3.11
./setup.sh
```

### No audio devices found
- **macOS**: Check System Preferences > Security & Privacy > Microphone
- **Linux**: Ensure your user is in the 'audio' group: `sudo usermod -a -G audio $USER`
- **All platforms**: List devices:
  ```python
  import sounddevice as sd
  print(sd.query_devices())
  ```

### Poor detection accuracy
- Ensure microphone is close to your mouth (within 6-12 inches)
- Try adjusting the threshold parameter
- Check that your microphone is working properly
- Minimize background noise

### Installation errors
- Ensure Python 3.11 or 3.12 is installed: `python3.11 --version`
- Update pip: `pip install --upgrade pip`
- Use automated setup: `./setup.sh` or `setup.bat`
- See [SETUP.md](SETUP.md) for detailed instructions

### Import errors
- Ensure you're in the correct directory
- Activate virtual environment: `source venv/bin/activate`
- Verify all dependencies are installed: `pip list`
- Try reinstalling: `pip install -r requirements.txt --force-reinstall`

For more troubleshooting help, see [SETUP.md](SETUP.md)

## Technical Details

### Audio Features Used

1. **Onset Strength**: Measures sudden amplitude changes
2. **Spectral Centroid**: Identifies frequency distribution
3. **RMS Energy**: Tracks energy envelope
4. **Zero-Crossing Rate**: Analyzes signal oscillations

### Detection Algorithm

1. Audio is captured in 100ms chunks
2. Each chunk is normalized
3. Multiple features are extracted
4. Features are weighted and combined
5. A confidence score is calculated
6. Detection triggered if score > threshold

## Future Enhancements

- [ ] Machine learning model for improved accuracy
- [ ] Multi-click pattern recognition
- [ ] GUI interface
- [ ] Mobile app integration
- [ ] Click-to-action mapping
- [ ] Training mode for personalized detection
- [ ] Support for other vocal sounds

## Dependencies

- `numpy`: Numerical computing
- `scipy`: Signal processing
- `sounddevice`: Audio I/O
- `librosa`: Audio feature extraction
- `numba`: Performance optimization
- `audioread`: Audio file reading
- `scikit-learn`: Machine learning utilities

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Better noise filtering
- Additional audio features
- Performance optimizations
- Cross-platform testing
- Documentation improvements

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the requirements.md for technical details
3. Test with the demo script to isolate problems

## Acknowledgments

Built using:
- [librosa](https://librosa.org/) for audio analysis
- [sounddevice](https://python-sounddevice.readthedocs.io/) for audio I/O
- Signal processing techniques from audio research community
