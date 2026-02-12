#!/bin/bash
# ============================================
# Tongue Click Detector - Complete Setup
# From fresh clone to running detector
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================"
echo " Tongue Click Detector - Full Setup"
echo "============================================"
echo ""
echo "Project: $PROJECT_DIR"
echo ""

# ------------------------------------------
# Step 1: Find best Python (3.11 preferred for numba/llvmlite compatibility)
# ------------------------------------------
echo "[1/5] Finding Python..."

PYTHON_CMD=""
for ver in python3.11 python3.12 python3.13 python3; do
    if command -v "$ver" &> /dev/null; then
        PYTHON_CMD="$ver"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3 not found."
    echo "Install with: brew install python@3.11"
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  Using: $PYTHON_CMD ($PY_VERSION)"
echo ""

# ------------------------------------------
# Step 2: Create virtual environment
# ------------------------------------------
echo "[2/5] Setting up virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Existing venv found. Removing..."
    rm -rf "$VENV_DIR"
fi

$PYTHON_CMD -m venv "$VENV_DIR"
echo "  venv created at: $VENV_DIR"
echo ""

# ------------------------------------------
# Step 3: Activate and upgrade pip
# ------------------------------------------
echo "[3/5] Activating venv and upgrading pip..."

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "  pip upgraded: $(pip --version)"
echo ""

# ------------------------------------------
# Step 4: Install dependencies
# ------------------------------------------
echo "[4/5] Installing dependencies..."
echo "  This may take a few minutes (librosa, numba, etc.)..."
echo ""

pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "  Installed packages:"
pip list --format=columns 2>/dev/null | head -25
echo ""

# ------------------------------------------
# Step 5: Verify installation
# ------------------------------------------
echo "[5/5] Verifying installation..."
echo ""

python "$PROJECT_DIR/verify_installation.py"

VERIFY_EXIT=$?

echo ""
echo "============================================"
if [ $VERIFY_EXIT -eq 0 ]; then
    echo " Setup Complete!"
else
    echo " Setup completed with warnings."
    echo " Check errors above."
fi
echo "============================================"
echo ""
echo "USAGE:"
echo ""
echo "  # Activate the venv (run this each time you open a new terminal)"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "  # Run the detector (basic)"
echo "  python tongue_click_detector.py"
echo ""
echo "  # Run the demo"
echo "  python demo.py"
echo ""
echo "  # Run the ML detector (after training)"
echo "  python ml_detector.py --mode realtime --duration 30"
echo ""
echo "  # Full ML training workflow"
echo "  ./quick_start.sh"
echo ""
echo "  # Deactivate when done"
echo "  deactivate"
echo ""
