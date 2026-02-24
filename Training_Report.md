venv) karthi@Mac tongue_click % python retrain_model.py \
    --positive training_data/positives \
    --negative training_data/auto_collected

======================================================================
LOADING DATASET
======================================================================

Loading positive samples from: training_data/positives
  Extracted 431 feature vectors

Loading negative samples from: training_data/auto_collected
  Extracted 183384 feature vectors

Positive samples: 431
Negative samples: 183384
Total samples: 183815
Feature dimensions: 26

======================================================================
TRAINING MODEL
======================================================================

Training set: 147052 samples
Test set: 36763 samples

Scaling features...

Training random_forest model...

======================================================================
EVALUATION
======================================================================

Training accuracy: 99.86%
Test accuracy: 99.81%
Cross-validation accuracy: 99.83% (+/- 0.01%)

Classification Report:
              precision    recall  f1-score   support

   Not Click       1.00      1.00      1.00     36677
Tongue Click       0.57      0.85      0.68        86

    accuracy                           1.00     36763
   macro avg       0.78      0.92      0.84     36763
weighted avg       1.00      1.00      1.00     36763


Confusion Matrix:
                 Predicted
              Not Click  Click
Actual Not Click   36621      56
       Click          13      73

Top 10 Most Important Features:
  1. spectral_bandwidth  : 0.2863
  2. rms_peak            : 0.1471
  3. rms_mean            : 0.0960
  4. mfcc_2              : 0.0812
  5. kurtosis            : 0.0656
  6. spectral_flatness   : 0.0555
  7. zcr                 : 0.0305
  8. onset_strength      : 0.0280
  9. spectral_flux       : 0.0269
  10. high_freq_ratio     : 0.0224

✓ Model saved to: models/tongue_click_model.pkl
✓ Scaler saved to: models/scaler.pkl

======================================================================
TRAINING COMPLETE!
======================================================================


But audio augmentation works great — creating slightly modified versions (pitch shift, speed change, add noise, volume change). These  
  are genuinely different to the model.                                                                                                  
                                                            
  Want me to create an augmentation script that takes your 40 samples and generates 2000+ by applying:
  - Pitch shifting (slightly higher/lower)
  - Time stretching (slightly faster/slower)
  - Adding background noise
  - Volume variation
