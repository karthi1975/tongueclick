# Tongue Click Detector - Complete Training Workflow

## Problem: False Positives from Household Sounds

Your current model detects tongue clicks but also triggers on:
- Dog barking
- Utensils (metal clanking)
- Sink/water sounds
- Bed creaking
- Scrubbing/cleaning noises
- Any other sharp/impulsive sounds

**Solution:** Retrain with negative examples so the model learns what tongue clicks are NOT.

---

## Complete Workflow

### Step 1: Collect Negative Examples (1-2 hours)

Collect 15-20 seconds of each false positive type:

```bash
# Interactive mode (recommended)
python collect_negative_samples.py --interactive

# Or specify category directly
python collect_negative_samples.py --category dog_bark --duration 15
python collect_negative_samples.py --category utensils_metal --duration 15
python collect_negative_samples.py --category sink_water --duration 15
python collect_negative_samples.py --category bed_creak --duration 15
python collect_negative_samples.py --category keyboard_typing --duration 15
```

**What to record:**
- **dog_bark**: Your dog barking (or find dog bark sounds)
- **utensils_metal**: Clanking plates, spoons, forks in sink
- **sink_water**: Tap running, water dripping, splashing
- **sink_cleaning**: Scrubbing dishes, sponge sounds
- **bed_creak**: Getting in/out of bed, shifting weight
- **keyboard_typing**: Typing, mouse clicks
- **door_close**: Closing doors, cabinets
- **ambient_noise**: General background noise
- **human_speech**: Speaking (especially sharp consonants: p, t, k)

**Target:** 100-200 total negative samples (10-20 per category)

```bash
# Check your progress
python collect_negative_samples.py --stats
```

---

### Step 2: Organize Your Data

**Directory structure:**
```
training_data/
├── positives/               # Your existing tongue click recordings
│   ├── click_001.wav
│   ├── click_002.wav
│   └── ...
└── negatives/               # Created by collect_negative_samples.py
    ├── dog_bark/
    │   ├── dog_bark_20260205_143022.wav
    │   └── ...
    ├── utensils_metal/
    ├── sink_water/
    ├── bed_creak/
    └── metadata.json        # Auto-generated
```

---

### Step 3: Train the Model

```bash
# Train with Random Forest (recommended - fast and accurate)
python retrain_model.py \
  --positive training_data/positives \
  --negative training_data/negatives \
  --model random_forest

# Or try Gradient Boosting (slower but sometimes better)
python retrain_model.py \
  --positive training_data/positives \
  --negative training_data/negatives \
  --model gradient_boost

# Or SVM (good for smaller datasets)
python retrain_model.py \
  --positive training_data/positives \
  --negative training_data/negatives \
  --model svm
```

**Output:**
```
LOADING DATASET
========================================================================
Loading positive samples from: training_data/positives
  Extracted 150 feature vectors
Loading negative samples from: training_data/negatives
  Extracted 200 feature vectors

Positive samples: 150
Negative samples: 200
Total samples: 350
Feature dimensions: 26

TRAINING MODEL
========================================================================
Training set: 280 samples
Test set: 70 samples

Training random_forest model...

EVALUATION
========================================================================
Training accuracy: 98.57%
Test accuracy: 95.71%
Cross-validation accuracy: 94.28% (+/- 2.31%)

Classification Report:
              precision    recall  f1-score   support
   Not Click       0.96      0.95      0.96        40
 Tongue Click       0.95      0.97      0.96        30

Confusion Matrix:
                 Predicted
              Not Click  Click
Actual Not Click     38      2
       Click          1     29

Top 10 Most Important Features:
  1. spectral_flatness    : 0.1523
  2. kurtosis            : 0.1287
  3. duration_ms         : 0.0945
  4. low_freq_ratio      : 0.0891
  5. hnr                 : 0.0756
  ...

✓ Model saved to: models/tongue_click_model.pkl
✓ Scaler saved to: models/scaler.pkl
```

**What to look for:**
- ✅ **Test accuracy >90%** - Good generalization
- ✅ **Similar precision/recall** - Balanced performance
- ✅ **Low false positives** (top-right in confusion matrix)
- ✅ **Low false negatives** (bottom-left in confusion matrix)

---

### Step 4: Test the New Model

#### Real-time testing:
```bash
# Default settings (70% confidence threshold)
python ml_detector.py --mode realtime --duration 30

# Stricter (fewer false positives, might miss softer clicks)
python ml_detector.py --mode realtime --duration 30 --threshold 0.85

# More sensitive (catches softer clicks, more false positives)
python ml_detector.py --mode realtime --duration 30 --threshold 0.60
```

#### File analysis:
```bash
python ml_detector.py --mode file --input test_recording.wav
```

**During testing:**
- Try making tongue clicks
- Try the sounds that previously caused false positives:
  - Talk near the microphone
  - Clank utensils
  - Type on keyboard
  - Close doors
  - etc.

---

### Step 5: Iterate if Needed

If you still get false positives:

1. **Collect MORE negative examples** of the specific sounds causing issues:
   ```bash
   # Example: if keyboard still triggers false positives
   python collect_negative_samples.py --category keyboard_typing --batch 10 --duration 3
   ```

2. **Retrain** with the expanded dataset:
   ```bash
   python retrain_model.py \
     --positive training_data/positives \
     --negative training_data/negatives
   ```

3. **Adjust threshold:**
   - If too many false positives → increase `--threshold` (e.g., 0.80, 0.85)
   - If missing real clicks → decrease `--threshold` (e.g., 0.60, 0.65)

---

## Advanced Features Explained

The new model uses **26 advanced features** to distinguish tongue clicks:

### Temporal Features (4)
- **duration_ms**: Tongue clicks are 10-50ms; barking/creaking are longer
- **attack_time_ms**: Clicks have instant attack (<5ms)
- **decay_time_ms**: Clicks decay fast (<30ms); utensils ring longer
- **kurtosis**: Clicks are very spiky (>10); speech is smoother (<5)

### Spectral Features (6)
- **spectral_centroid**: Frequency center; clicks are high (2500-4000Hz)
- **spectral_bandwidth**: Frequency spread; clicks are broadband
- **spectral_rolloff**: High-frequency cutoff; clicks >4000Hz
- **spectral_flatness**: Clicks are noise-like (0.8-0.95); barks are tonal (<0.6)
- **spectral_flux**: Spectral stability; clicks are stable; utensils vary
- **spectral_centroid_variance**: Frequency consistency

### Energy Features (4)
- **rms_peak/mean**: Energy distribution
- **peak_to_mean_ratio**: Clicks are very impulsive (>2.0)
- **onset_strength**: Transient sharpness

### Harmonic Features (3)
- **harmonic_to_noise_ratio**: Clicks have no pitch (low HNR)
- **has_pitch**: Clicks have no pitch; barks do
- **zero_crossing_rate**: Signal oscillation rate

### Frequency Distribution (3)
- **low/mid/high_freq_ratio**: Energy distribution across frequency bands
- Clicks have minimal low-freq energy (<25%)

### Decay Characteristics (2)
- **has_ringing**: Metallic sounds ring; clicks don't
- **decay_type**: 'fast' for clicks, 'ringing' for utensils, 'slow' for creaks

### Timbre (MFCCs) (5)
- **mfcc_1-5**: Capture the "color" of the sound

---

## Troubleshooting

### Problem: "No samples loaded"
- Check directory paths are correct
- Make sure .wav files exist in `training_data/positives/` and `training_data/negatives/`

### Problem: "Test accuracy <80%"
- Collect more training data (aim for 200+ samples each class)
- Check if data quality is good (no clipped audio, proper recording)
- Try different model types

### Problem: "Model always predicts one class"
- Dataset is imbalanced (need equal positive and negative samples)
- Collect more samples of the underrepresented class

### Problem: "Import errors"
- Install dependencies:
  ```bash
  pip install numpy scipy librosa sounddevice soundfile scikit-learn joblib
  ```

---

## Performance Tuning

### Confidence Threshold Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.50-0.65 | Very sensitive | Quiet environment, soft clicks |
| 0.70-0.75 | Balanced (recommended) | Normal use |
| 0.80-0.90 | Very strict | Noisy environment |
| 0.95+ | Extremely strict | Only very clear clicks |

### Model Selection Guide

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| Random Forest | Fast | High | Medium | General use (recommended) |
| Gradient Boost | Medium | Very High | Medium | Best accuracy |
| SVM | Slow | High | Low | Small datasets |

---

## Integration with Your Application

Once trained, integrate the ML detector:

```python
from ml_detector import MLTongueClickDetector

# Initialize
detector = MLTongueClickDetector(
    model_path='models/tongue_click_model.pkl',
    scaler_path='models/scaler.pkl',
    confidence_threshold=0.75
)

# Real-time detection with callback
def on_click_detected(timestamp, confidence):
    print(f"Click at {timestamp:.2f}s with {confidence:.0%} confidence")
    # Your application logic here

detections = detector.real_time_detection(
    duration=60,
    callback=on_click_detected
)
```

---

## Expected Results

With proper training data:

- ✅ **95%+ accuracy** on test set
- ✅ **<5% false positive rate** (1-2 false clicks per minute in noisy environment)
- ✅ **<5% false negative rate** (catches 95%+ of real clicks)
- ✅ **Robust to household sounds** (dog barking, utensils, etc.)

---

## Next Steps

1. Start collecting negative samples (1-2 hours)
2. Train initial model
3. Test and iterate
4. Fine-tune confidence threshold
5. Integrate into your application

Good luck! 🎯
