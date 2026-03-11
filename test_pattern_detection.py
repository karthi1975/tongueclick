#!/usr/bin/env python3
"""Tests for the confirmation pattern detection logic in CallForAttention."""

import sys
import time
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Mock hardware-dependent modules before importing call_for_attention
sys.modules['sounddevice'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['librosa.feature'] = MagicMock()
sys.modules['librosa.onset'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()

# Mock joblib before import
mock_joblib = MagicMock()
mock_joblib.load.return_value = MagicMock()
sys.modules['joblib'] = mock_joblib

# Mock advanced_features
mock_extractor = MagicMock()
mock_af_module = MagicMock()
mock_af_module.AdvancedFeatureExtractor.return_value = mock_extractor
sys.modules['advanced_features'] = mock_af_module

from call_for_attention import CallForAttention


class TestPatternDetection(unittest.TestCase):
    """Test the 3-pause-3 confirmation pattern with rhythm validation."""

    def _make_listener(self, **kwargs):
        """Create a CallForAttention instance with mocked model/scaler."""
        defaults = dict(
            model_path='models/tongue_click_model.pkl',
            scaler_path='models/scaler.pkl',
            sample_rate=44100,
            confidence_threshold=0.93,
            clicks_group1=2,
            clicks_group2=3,
            group_timeout=3.0,
            pause_min=0.2,
            pause_max=5.0,
            rhythm_max_cv=0.8,
            debounce_interval=0.15,
            save_clicks=False,
        )
        defaults.update(kwargs)

        listener = CallForAttention(**defaults)
        return listener

    AUDIO = np.random.randn(4410).astype(np.float32) * 0.1

    def test_initial_state(self):
        listener = self._make_listener()
        self.assertEqual(listener.state, 'waiting_first_group')
        self.assertEqual(listener.click_times, [])
        self.assertEqual(listener.total_triggers, 0)

    def test_rhythm_check_regular(self):
        """Evenly spaced clicks should pass rhythm check."""
        listener = self._make_listener()
        times = [0.0, 0.5, 1.0]
        self.assertTrue(listener._check_rhythm(times))

    def test_rhythm_check_irregular(self):
        """Irregularly spaced clicks should fail rhythm check."""
        listener = self._make_listener(rhythm_max_cv=0.3)
        times = [0.0, 0.1, 1.5]
        self.assertFalse(listener._check_rhythm(times))

    def test_rhythm_check_too_fast(self):
        """Clicks faster than 0.05s apart should fail."""
        listener = self._make_listener()
        times = [0.0, 0.02, 0.04]
        self.assertFalse(listener._check_rhythm(times))

    def test_rhythm_check_single_click(self):
        """Single click should always pass (not enough data to judge)."""
        listener = self._make_listener()
        self.assertTrue(listener._check_rhythm([1.0]))

    def test_full_pattern_triggers_webhook(self):
        """2 clicks, pause, 3 clicks should trigger webhook."""
        listener = self._make_listener()

        with patch.object(listener, '_trigger_webhook') as mock_trigger, \
             patch.object(listener, '_save_audio', return_value=None):

            base = time.time()

            # Group 1: 2 clicks
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)  # click 1/2
            with patch('time.time', return_value=base + 0.5):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/2

            self.assertEqual(listener.state, 'waiting_pause')

            # Pause (1.0s) then Group 2: 3 clicks
            with patch('time.time', return_value=base + 1.5):
                listener.last_click_time = base + 0.5
                listener._on_click_detected(self.AUDIO, 0.94)  # click 1/3
            self.assertEqual(listener.state, 'waiting_second_group')

            with patch('time.time', return_value=base + 2.0):
                listener.last_click_time = base + 1.5
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/3
            with patch('time.time', return_value=base + 2.5):
                listener.last_click_time = base + 2.0
                listener._on_click_detected(self.AUDIO, 0.95)  # click 3/3

            mock_trigger.assert_called_once()
            self.assertEqual(listener.state, 'waiting_first_group')

    def test_pause_too_short_resets(self):
        """Click arriving too soon after group 1 should reset."""
        listener = self._make_listener()

        with patch.object(listener, '_save_audio', return_value=None):
            base = time.time()

            # Group 1: 2 clicks
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)  # click 1/2
            with patch('time.time', return_value=base + 0.5):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/2

            self.assertEqual(listener.state, 'waiting_pause')

            # Click too soon (0.16s pause < pause_min 0.2s, past debounce 0.15s)
            with patch('time.time', return_value=base + 0.66):
                listener.last_click_time = base + 0.5
                listener._on_click_detected(self.AUDIO, 0.94)

            self.assertEqual(listener.state, 'waiting_first_group')

    def test_pause_too_long_resets(self):
        """Timeout during pause should reset state."""
        listener = self._make_listener(pause_max=5.0)

        with patch.object(listener, '_save_audio', return_value=None):
            base = time.time()

            # Group 1: 2 clicks
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)  # click 1/2
            with patch('time.time', return_value=base + 0.5):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/2

            self.assertEqual(listener.state, 'waiting_pause')
            listener.first_group_end_time = base + 0.5

            # Timeout check after 6s (> pause_max 5.0s)
            with patch('time.time', return_value=base + 7.0):
                listener._check_timeout()

            self.assertEqual(listener.state, 'waiting_first_group')

    def test_group1_timeout_resets(self):
        """If too long between clicks in group 1, reset."""
        listener = self._make_listener()

        with patch.object(listener, '_save_audio', return_value=None):
            base = time.time()

            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)

            self.assertEqual(len(listener.click_times), 1)

            # 4s passes (> group_timeout 3.0s)
            with patch('time.time', return_value=base + 4.0):
                listener._check_timeout()

            self.assertEqual(listener.state, 'waiting_first_group')
            self.assertEqual(listener.click_times, [])

    def test_irregular_rhythm_group1_resets(self):
        """Irregular rhythm in group 1 should reject and reset."""
        # Use 3 clicks in group 1 with strict CV to test rhythm rejection
        listener = self._make_listener(clicks_group1=3, rhythm_max_cv=0.3)

        with patch.object(listener, '_save_audio', return_value=None):
            base = time.time()

            # Very uneven intervals: 0.2s then 1.2s
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)
            with patch('time.time', return_value=base + 0.2):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)
            with patch('time.time', return_value=base + 1.4):
                listener.last_click_time = base + 0.2
                listener._on_click_detected(self.AUDIO, 0.95)

            self.assertEqual(listener.state, 'waiting_first_group')
            self.assertEqual(listener.false_rhythm_count, 1)

    def test_debounce_ignores_rapid_clicks(self):
        """Clicks within debounce interval should be ignored."""
        listener = self._make_listener()

        with patch.object(listener, '_save_audio', return_value=None):
            base = time.time()

            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)

            # Click 0.05s later (within debounce)
            with patch('time.time', return_value=base + 0.05):
                listener._on_click_detected(self.AUDIO, 0.95)

            self.assertEqual(len(listener.click_times), 1)
            self.assertEqual(listener.total_clicks, 1)

    def test_no_trigger_without_full_pattern(self):
        """Only group 1 without group 2 should never trigger."""
        listener = self._make_listener()

        with patch.object(listener, '_trigger_webhook') as mock_trigger, \
             patch.object(listener, '_save_audio', return_value=None):

            base = time.time()

            # Complete group 1 only (2 clicks)
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)  # click 1/2
            with patch('time.time', return_value=base + 0.5):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/2

            mock_trigger.assert_not_called()

    def test_long_pause_still_works(self):
        """Becca may need up to 5s pause - should still work."""
        listener = self._make_listener()

        with patch.object(listener, '_trigger_webhook') as mock_trigger, \
             patch.object(listener, '_save_audio', return_value=None):

            base = time.time()

            # Group 1: 2 clicks
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)  # click 1/2
            with patch('time.time', return_value=base + 0.5):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/2

            self.assertEqual(listener.state, 'waiting_pause')

            # 4 second pause (within 5.0s max)
            with patch('time.time', return_value=base + 4.5):
                listener.last_click_time = base + 0.5
                listener._on_click_detected(self.AUDIO, 0.94)  # click 1/3
            self.assertEqual(listener.state, 'waiting_second_group')

            with patch('time.time', return_value=base + 5.0):
                listener.last_click_time = base + 4.5
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/3
            with patch('time.time', return_value=base + 5.5):
                listener.last_click_time = base + 5.0
                listener._on_click_detected(self.AUDIO, 0.95)  # click 3/3

            mock_trigger.assert_called_once()

    def test_pause_too_short_resets_from_pause_state(self):
        """Clicking too soon after group 1 resets even though click_times is empty."""
        listener = self._make_listener()

        with patch.object(listener, '_save_audio', return_value=None):
            base = time.time()

            # Complete group 1: 2 clicks
            with patch('time.time', return_value=base):
                listener._on_click_detected(self.AUDIO, 0.95)  # click 1/2
            with patch('time.time', return_value=base + 0.5):
                listener.last_click_time = base
                listener._on_click_detected(self.AUDIO, 0.95)  # click 2/2

            self.assertEqual(listener.state, 'waiting_pause')
            # click_times is empty after entering pause state
            self.assertEqual(listener.click_times, [])

            # Click too soon (0.15s pause < pause_min 0.2s, but past debounce)
            with patch('time.time', return_value=base + 0.65):
                listener.last_click_time = base + 0.5
                listener._on_click_detected(self.AUDIO, 0.94)

            # Must reset back to first group, not get stuck
            self.assertEqual(listener.state, 'waiting_first_group')


if __name__ == '__main__':
    unittest.main()
