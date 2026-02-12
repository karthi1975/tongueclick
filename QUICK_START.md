# Quick Start Guide

## TL;DR - Get Started in 30 Seconds

### macOS/Linux
```bash
./setup.sh
python tongue_click_detector.py
```

### Windows
```cmd
setup.bat
python tongue_click_detector.py
```

---

## What You Need

- Python 3.11 or 3.12 (NOT 3.13)
- A microphone

## Check Python Version

```bash
python3.11 --version
```

If you don't have it:
- **macOS**: `brew install python@3.11`
- **Ubuntu**: `sudo apt-get install python3.11`
- **Windows**: Download from python.org

## Installation Options

### Option 1: Automated (Recommended)

**macOS/Linux:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Option 2: Manual

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run the Detector

```bash
python tongue_click_detector.py
```

Make tongue click sounds when prompted!

## Common Issues

### "OSError: Could not find/load shared object file"
➜ You're using Python 3.13. Use Python 3.11 or 3.12 instead.

### "No module named 'librosa'"
➜ Activate the virtual environment: `source venv/bin/activate`

### No clicks detected
➜ Make louder clicks closer to the microphone

## More Help

- Detailed setup: [SETUP.md](SETUP.md)
- Full documentation: [README.md](README.md)
- Code reference: See comments in `tongue_click_detector.py`

## File Overview

| File | Purpose |
|------|---------|
| `setup.sh` | Automated setup for macOS/Linux |
| `setup.bat` | Automated setup for Windows |
| `requirements.txt` | Python dependencies |
| `tongue_click_detector.py` | Main detector code |
| `README.md` | Full documentation |
| `SETUP.md` | Detailed setup guide |
| `QUICK_START.md` | This file |
