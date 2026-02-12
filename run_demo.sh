#!/bin/bash
# Quick script to run the demo with virtual environment

# Activate venv and run demo
source venv/bin/activate
python demo.py "$@"
deactivate
