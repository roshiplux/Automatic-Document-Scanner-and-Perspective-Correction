#!/bin/bash

# Quick Live Scanner Script
# Automatically activates venv and launches live camera scanner

cd "$(dirname "$0")"

echo "🚀 Starting Live Document Scanner..."
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install opencv-python numpy"
    exit 1
fi

# Run the live scanner
cd src
python phone_camera_guide.py
