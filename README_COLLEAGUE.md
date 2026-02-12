# Tongue Click Detector

Real-time tongue click sound detection using Python.

## Quick Setup

### Requirements
- Python 3.11 or 3.12 (NOT 3.13)

### Installation

**Mac/Linux:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Run
```bash
python tongue_click_detector.py
```

Make tongue click sounds when prompted!

## Manual Setup (if scripts don't work)

```bash
# 1. Create virtual environment
python3.11 -m venv venv

# 2. Activate
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python tongue_click_detector.py
```

## Troubleshooting

**"OSError: Could not find/load shared object"**
→ Use Python 3.11 or 3.12, NOT 3.13

**"No module named librosa"**
→ Activate virtual environment: `source venv/bin/activate`

**Detector picks up speech/conversations**
→ **FIXED!** Now uses strict filtering (3kHz+, 4x impulsive ratio)

**No clicks detected**
→ Click louder, closer to microphone

**Missing some real clicks**
→ See "Tuning" section below

## Tuning (If Needed)

The detector is **fine-tuned to ONLY detect tongue clicks** and filter out speech.

**Default settings (filters speech):**
```python
threshold=0.5                 # High: Sharp onsets only
confidence_threshold=0.85     # High: Very strict
min_spectral_centroid=3000    # High freq: Tongue clicks 3kHz+
min_peak_to_mean_ratio=4.0    # Very impulsive
```

**If missing real clicks**, edit line 518 in `tongue_click_detector.py`:
```python
config = DetectorConfig(
    threshold=0.4,                    # Lower = more sensitive
    confidence_threshold=0.75,        # Lower = less strict
    min_spectral_centroid=2500,       # Lower = catch quieter clicks
    min_peak_to_mean_ratio=3.5        # Lower = less impulsive
)
```

**If still detecting speech**, increase all values:
```python
threshold=0.6, confidence_threshold=0.90, min_spectral_centroid=3500
```

## Why It Filters Speech Now

| Feature | Tongue Click | Speech | Filter |
|---------|--------------|--------|--------|
| Frequency | 3000-8000 Hz | 100-3000 Hz | ✅ Too low |
| Duration | 20-50 ms | 50-300+ ms | ✅ Too long |
| Onset | Very sharp | Moderate | ✅ Too gradual |
| Peak/Mean | 5-10x | 2-3x | ✅ Not impulsive |

## Questions?

Check the code - it's well-documented with SOLID principles.
