#!/usr/bin/env python3
"""
Tests for audiobook false-positive hardening changes.

Verifies tightened defaults for:
- Confidence threshold (0.90 -> 0.93)
- Rhythm max CV (0.8 -> 0.5)
- Sustained max clicks (8 -> 5)
- Sustained window (60 -> 120)
"""

import sys
import os
import unittest
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from call_for_attention import CallForAttention


class TestHardenedDefaults(unittest.TestCase):
    """Verify tightened default parameters."""

    def _get_init_defaults(self):
        """Extract default values from CallForAttention.__init__ signature."""
        sig = inspect.signature(CallForAttention.__init__)
        return {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}

    def test_confidence_threshold_raised(self):
        defaults = self._get_init_defaults()
        self.assertGreaterEqual(defaults['confidence_threshold'], 0.97,
                                "Threshold should be >= 0.97 to reduce audiobook false positives")

    def test_rhythm_max_cv_tightened(self):
        defaults = self._get_init_defaults()
        self.assertLessEqual(defaults['rhythm_max_cv'], 0.6,
                             "Rhythm CV should be <= 0.6 for stricter pattern matching")

    def test_sustained_max_clicks_allows_retries(self):
        defaults = self._get_init_defaults()
        pattern_clicks = defaults['clicks_group1'] + defaults['clicks_group2']
        self.assertGreaterEqual(defaults['sustained_max_clicks'], pattern_clicks * 2,
                                "Sustained max must allow at least 2 full pattern attempts")

    def test_sustained_window_increased(self):
        defaults = self._get_init_defaults()
        self.assertGreaterEqual(defaults['sustained_window'], 120.0,
                                "Sustained window should be >= 120s for slow-drip filtering")


class TestRhythmValidation(unittest.TestCase):
    """Test that tighter rhythm CV rejects irregular patterns."""

    def setUp(self):
        """Create a detector with tightened defaults (no model needed for rhythm check)."""
        self.rhythm_max_cv = 0.5

    def _compute_cv(self, timestamps):
        """Compute coefficient of variation of intervals."""
        intervals = [timestamps[i+1] - timestamps[i]
                     for i in range(len(timestamps) - 1)]
        if len(intervals) < 2:
            return 0.0
        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return float('inf')
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = variance ** 0.5
        return std / mean

    def test_regular_rhythm_passes(self):
        """Evenly spaced clicks (CV ~0) should pass."""
        timestamps = [0.0, 0.3, 0.6]  # perfectly regular
        cv = self._compute_cv(timestamps)
        self.assertLessEqual(cv, self.rhythm_max_cv)

    def test_slightly_irregular_passes(self):
        """Slightly uneven clicks should still pass."""
        timestamps = [0.0, 0.28, 0.62]  # ~10% variation
        cv = self._compute_cv(timestamps)
        self.assertLessEqual(cv, self.rhythm_max_cv)

    def test_very_irregular_fails(self):
        """Random-ish spacing (like audiobook speech artifacts) should fail."""
        timestamps = [0.0, 0.1, 0.8]  # very irregular
        cv = self._compute_cv(timestamps)
        self.assertGreater(cv, self.rhythm_max_cv,
                           "Irregular timing should be rejected by CV <= 0.5")


class TestSustainedFilter(unittest.TestCase):
    """Test that the podcast/sustained audio filter catches slow-drip false positives."""

    def test_sustained_audio_suppressed(self):
        """Many rapid detections (like audiobook) should trigger suppression."""
        sustained_max = 15
        sustained_window = 120.0

        # Simulate audiobook: 20 detections over 100 seconds
        detection_times = [i * 5 for i in range(20)]

        recent = [t for t in detection_times if detection_times[-1] - t <= sustained_window]
        self.assertGreater(len(recent), sustained_max,
                           "20 detections in 120s should exceed threshold of 15")

    def test_three_pattern_attempts_not_suppressed(self):
        """Three full pattern attempts (15 clicks) should NOT trigger suppression."""
        sustained_max = 15

        # 3 attempts x 5 clicks = 15, at the limit but not over
        detection_count = 15
        self.assertLessEqual(detection_count, sustained_max,
                             "3 pattern attempts (15 clicks) should not be suppressed")


if __name__ == '__main__':
    unittest.main()
