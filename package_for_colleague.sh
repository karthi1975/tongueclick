#!/bin/bash
# Package tongue click detector for sharing with colleagues

echo "Creating package for colleague..."

# Create package directory
PACKAGE_DIR="tongue_click_package"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Copy essential files
echo "Copying files..."
cp tongue_click_detector.py "$PACKAGE_DIR/"
cp requirements.txt "$PACKAGE_DIR/"
cp setup.sh "$PACKAGE_DIR/"
cp setup.bat "$PACKAGE_DIR/"
cp README_COLLEAGUE.md "$PACKAGE_DIR/README.md"

# Make setup scripts executable
chmod +x "$PACKAGE_DIR/setup.sh"

echo "✓ Package created: $PACKAGE_DIR/"
echo ""
echo "Files included:"
ls -lh "$PACKAGE_DIR"
echo ""
echo "To share:"
echo "  1. Compress: zip -r tongue_click_package.zip $PACKAGE_DIR"
echo "  2. Or upload the folder to Git/Cloud"
echo ""
