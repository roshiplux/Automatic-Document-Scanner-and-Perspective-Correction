# Automatic Document Scanner and Perspective Correction

AI-powered document scanner that uses mobile phone cameras for high-quality scanning with automatic edge detection, perspective correction, and real-time visual guidance.

## Features

- **Real-time Document Detection** - 6 detection strategies for accurate edge finding
- **Mobile Camera Support** - Works with iPhone and Android cameras
- **Visual Guidance System** - Real-time feedback with movement tracking
- **Auto-Capture** - Smart stability detection for perfect shots
- **Multiple Enhancement Modes** - Color, B&W, and adaptive processing
- **4-Point Perspective Correction** - Automatic document straightening

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start scanning
./live_scan.sh
```

## Usage Modes

1. **Live Camera Scanning** - Real-time scanning with phone camera
2. **GUI Mode** - User-friendly graphical interface
3. **CLI Mode** - Command-line batch processing
4. **Photo Mode** - Process existing photos

For detailed instructions, see [HOW_TO_START_AND_USE.txt](HOW_TO_START_AND_USE.txt)

## Requirements

- Python 3.7+
- OpenCV, NumPy, imutils
- Mobile phone camera (iPhone or Android)

## Camera Setup

- **iPhone**: Continuity Camera (iOS 16+) or EpocCam/iVCam
- **Android**: DroidCam or IP Webcam

## Output

Scanned documents are saved to `output/` folder in high-quality JPEG format.

## License

MIT License

## Author

[Your Name]
