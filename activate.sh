#!/bin/bash
# Helper script to activate the virtual environment and provide usage instructions

echo "================================================"
echo "Tongue Click Detector - Virtual Environment"
echo "================================================"
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated!"
echo ""
echo "Available commands:"
echo "  python demo.py                     # Interactive demo"
echo "  python demo.py --mode realtime     # Real-time detection"
echo "  python demo.py --mode devices      # List audio devices"
echo "  python test_basic.py               # Run tests"
echo ""
echo "To deactivate: type 'deactivate'"
echo "================================================"
echo ""

# Start a new shell with the venv activated
exec $SHELL
