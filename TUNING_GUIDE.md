# Tongue Click Detector - Tuning Guide

## Problem: Detector picks up speech/conversations

The detector is now **fine-tuned to ONLY detect tongue clicks** and filter out:
- ❌ Speech and conversations
- ❌ Background noise
- ❌ Keyboard typing
- ❌ Other sounds

## What Changed

### Stricter Default Settings

```python
config = DetectorConfig(
    threshold=0.5,                    # Was 0.3 → Now STRICTER
    confidence_threshold=0.85,        # Was 0.6 → Now STRICTER
    min_spectral_centroid=3000,       # NEW: Tongue clicks are 3kHz+
    min_peak_to_mean_ratio=4.0,       # NEW: Must be very impulsive
    min_energy_threshold=0.02         # Increased from 0.01
)
```

### Detection Logic Changed

**Old logic:** If weighted score > threshold (could pick up speech)

**New logic:** Must meet ALL of these:
- ✅ Very sharp onset (threshold > 0.5)
- ✅ Very high frequency (> 3000 Hz)
- ✅ Very impulsive (peak/mean ratio > 4x)
- ✅ High confidence score (> 0.85)

This filters out speech because:
- Speech frequency: 100-3000 Hz (too low)
- Speech envelope: Smooth, not impulsive
- Speech duration: Longer than clicks

## How to Tune for Your Environment

### If Missing Real Clicks (Too Strict)

Make it **less strict**:

```python
config = DetectorConfig(
    threshold=0.4,                    # Lower (was 0.5)
    confidence_threshold=0.75,        # Lower (was 0.85)
    min_spectral_centroid=2500,       # Lower (was 3000)
    min_peak_to_mean_ratio=3.5        # Lower (was 4.0)
)
```

### If Still Picking Up Speech (Too Lenient)

Make it **more strict**:

```python
config = DetectorConfig(
    threshold=0.6,                    # Higher
    confidence_threshold=0.90,        # Higher
    min_spectral_centroid=3500,       # Higher
    min_peak_to_mean_ratio=5.0        # Higher
)
```

### Testing Mode (See All Detections)

To debug and see what's being detected:

```python
# LENIENT - Will detect most sounds
config = DetectorConfig(
    threshold=0.3,
    confidence_threshold=0.6,
    min_spectral_centroid=2000,
    min_peak_to_mean_ratio=3.0
)
```

## Parameter Explanation

| Parameter | What It Does | Effect on Speech |
|-----------|--------------|------------------|
| **threshold** | Onset sharpness (0-1) | Higher = filters gradual onsets (vowels) |
| **confidence_threshold** | Overall confidence (0-1) | Higher = stricter detection |
| **min_spectral_centroid** | Minimum frequency (Hz) | Higher = filters low-freq speech |
| **min_peak_to_mean_ratio** | Impulsiveness (ratio) | Higher = filters smooth speech |
| **min_energy_threshold** | Minimum volume | Higher = ignores quiet sounds |

## Quick Presets

### 1. VERY STRICT (Noisy Office)
```python
# Use when lots of background noise/conversation
config = DetectorConfig(
    threshold=0.6,
    confidence_threshold=0.90,
    min_spectral_centroid=3500,
    min_peak_to_mean_ratio=5.0,
    min_energy_threshold=0.03
)
```

### 2. BALANCED (Default)
```python
# Good for most environments
config = DetectorConfig(
    threshold=0.5,
    confidence_threshold=0.85,
    min_spectral_centroid=3000,
    min_peak_to_mean_ratio=4.0,
    min_energy_threshold=0.02
)
```

### 3. SENSITIVE (Quiet Room)
```python
# Use in quiet environment, need to catch soft clicks
config = DetectorConfig(
    threshold=0.4,
    confidence_threshold=0.75,
    min_spectral_centroid=2500,
    min_peak_to_mean_ratio=3.5,
    min_energy_threshold=0.01
)
```

## How to Change Settings

Edit `tongue_click_detector.py` around line 518:

```python
if __name__ == "__main__":
    # Change these values ↓
    config = DetectorConfig(
        threshold=0.5,                    # Adjust this
        confidence_threshold=0.85,        # Adjust this
        min_spectral_centroid=3000,       # Adjust this
        min_peak_to_mean_ratio=4.0        # Adjust this
    )
```

Then run: `python tongue_click_detector.py`

## Troubleshooting

### Still Detecting Speech

1. **Increase** `min_spectral_centroid` to 3500+
2. **Increase** `min_peak_to_mean_ratio` to 5.0+
3. **Increase** `threshold` to 0.6+

### Missing Tongue Clicks

1. **Decrease** `confidence_threshold` to 0.75
2. **Decrease** `threshold` to 0.4
3. Check microphone distance (should be 6-12 inches)
4. Make louder clicks

### Detecting Keyboard/Typing

1. **Increase** `min_spectral_centroid` (keyboard is lower freq)
2. **Increase** `min_peak_to_mean_ratio` (keyboard less impulsive)

## Technical Details

### Why Speech is Filtered Out

| Feature | Tongue Click | Speech | Result |
|---------|--------------|--------|--------|
| Frequency | 3000-8000 Hz | 100-3000 Hz | ✅ Filtered by min_spectral_centroid |
| Duration | 20-50 ms | 50-300 ms | ✅ Filtered by impulsiveness |
| Onset | Very sharp | Moderate | ✅ Filtered by threshold |
| Peak/Mean | 5-10x | 2-3x | ✅ Filtered by min_peak_to_mean_ratio |

### Feature Weights

The confidence score is calculated as:
```
score = (onset * 0.5) + (frequency * 0.3) + (impulsive * 0.2)
```

Onset is weighted highest because it's the most distinctive feature of clicks.

## Testing Your Configuration

1. Run the detector
2. Make a tongue click
3. Try speaking normally
4. Results:
   - ✅ **Good**: Detects clicks, ignores speech
   - ❌ **Too strict**: Misses clicks → Lower thresholds
   - ❌ **Too lenient**: Detects speech → Raise thresholds

## Need Help?

The code includes detailed comments explaining each parameter at:
- Line 64-75: `DetectorConfig` class
- Line 208-250: Classification logic
- Line 517-533: Example configuration
