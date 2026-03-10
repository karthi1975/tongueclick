#!/usr/bin/env python3
"""
Wrapper to run call_for_attention with warnings suppressed.
Use this on Raspberry Pi to avoid sklearn warning spam.

Usage:
    python call_for_help.py --threshold 0.93 --device 1 --sample-rate 16000 --model-dir models_16k
"""

import warnings
warnings.filterwarnings("ignore")

from call_for_attention import main

if __name__ == "__main__":
    main()
