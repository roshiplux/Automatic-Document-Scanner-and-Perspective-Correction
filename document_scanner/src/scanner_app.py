"""
Document Scanner Application
Main application with interactive GUI for automatic document scanning and perspective correction.
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

from document_detector import DocumentDetector
from perspective_corrector import PerspectiveCorrector


class DocumentScanner:
    """Main application class for document scanning."""
    
    def __init__(self):
        """Initialize the document scanner."""
        # Use more sensitive detection settings
        self.detector = DocumentDetector(min_area_ratio=0.05, max_area_ratio=0.95)
        self.corrector = PerspectiveCorrector()
        self.current_image = None
        self.corners = None
        self.mode = 'webcam'  # 'webcam' or 'file'
    
    def _draw_camera_guidance(self, frame, corners_detected=False, stable=False, movement_amount=0):
        """Draw real-time guidance overlay for optimal camera positioning."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw center crosshair
        center_x, center_y = w // 2, h // 2
        cv2.line(overlay, (center_x - 40, center_y), (center_x + 40, center_y), (255, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y - 40), (center_x, center_y + 40), (255, 255, 255), 2)
        
        # Draw target rectangle for document placement
        margin = int(min(w, h) * 0.1)
        cv2.rectangle(overlay, (margin, margin), (w - margin, h - margin), (150, 150, 150), 2)
        
        # Status indicator box
        if corners_detected and stable:
            status_text = "✓ PERFECT - PRESS SPACE"
            status_color = (0, 255, 0)
            box_color = (0, 200, 0)
        elif corners_detected:
            status_text = "HOLD STEADY"
            status_color = (0, 255, 255)
            box_color = (0, 200, 200)
        else:
            status_text = "POSITION DOCUMENT"
            status_color = (0, 165, 255)
            box_color = (0, 130, 200)
        
        # Draw status box at top
        cv2.rectangle(overlay, (10, 10), (380, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (380, 80), box_color, 3)
        cv2.putText(overlay, status_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Movement indicator
        if corners_detected:
            if movement_amount < 10:
                move_text = "Steady!"
                move_color = (0, 255, 0)
            elif movement_amount < 20:
                move_text = "Small movement"
                move_color = (0, 255, 255)
            else:
                move_text = "Too much movement"
                move_color = (0, 100, 255)
            cv2.putText(overlay, move_text, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, move_color, 2)
        
        # Draw guidance tips at bottom
        tips = [
            "📱 Hold phone DIRECTLY above (90°)",
            "📐 Keep parallel to document",
            "💡 Avoid shadows & glare",
            "🎯 Center document in frame"
        ]
        
        y_offset = h - 130
        cv2.rectangle(overlay, (5, y_offset - 35), (w - 5, h - 5), (0, 0, 0), -1)
        for i, tip in enumerate(tips):
            cv2.putText(overlay, tip, (15, y_offset + i * 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Blend overlay
        return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        
    def scan_from_webcam(self, camera_index=None):
        """Interactive mobile camera scanning mode."""
        print("\n=== Mobile Camera Scanner Mode ===")
        
        # Auto-detect mobile cameras if not specified
        if camera_index is None:
            camera_index = self._select_camera()
        
        print("Controls:")
        print("  SPACE - Capture and scan document")
        print("  'q'   - Quit")
        print("  'r'   - Reset/clear current scan")
        print("  'e'   - Show edge detection (debug)")
        print("\n📸 TIPS FOR BEST RESULTS:")
        print("  • Place paper on a DARK or contrasting surface")
        print("  • Ensure GOOD LIGHTING (no shadows)")
        print("  • Keep paper FLAT and fully visible")
        print("  • Paper should fill 20-80% of frame")
        print("  • Wait for STABLE indicator before capturing")
        print("\n")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        captured_image = None
        detected_corners = None
        show_edges = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # If we have a captured image, show it
            if captured_image is not None:
                display = captured_image.copy()
                
                # Show instructions
                cv2.putText(display, "Press 's' to save | 'e' to enhance | 'r' to recapture", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Document Scanner', display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # Save the scanned document
                    if detected_corners is not None:
                        self._save_scanned_document(frame, detected_corners)
                    captured_image = None
                    detected_corners = None
                    
                elif key == ord('e'):
                    # Show enhancement options
                    if detected_corners is not None:
                        self._show_enhancement_options(frame, detected_corners)
                    captured_image = None
                    detected_corners = None
                    
                elif key == ord('r'):
                    # Reset and go back to live view
                    captured_image = None
                    detected_corners = None
                    
                elif key == ord('q'):
                    break
            else:
                # Live view with detection
                success, corners, vis_image = self.detector.detect(frame)
                
                # Show edge detection if requested
                if show_edges:
                    gray = self.detector.preprocess_image(frame)
                    edges = self.detector.detect_edges(gray)
                    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    # Resize to match main view height if needed
                    h, w = vis_image.shape[:2]
                    edges_resized = cv2.resize(edges_color, (w // 2, h))
                    vis_image_resized = cv2.resize(vis_image, (w // 2, h))
                    vis_image = np.hstack([vis_image_resized, edges_resized])
                
                # Add enhanced status overlay
                status_color = (0, 255, 0) if success else (0, 0, 255)
                
                # Add confidence indicator if detected
                if success:
                    confidence = self.detector.stable_confidence
                    status_text = f"DOCUMENT DETECTED - Stability: {confidence}/10"
                    # Show green indicator when stable
                    if confidence >= 7:
                        cv2.rectangle(vis_image, (10, 80), (30, 100), (0, 255, 0), -1)
                        cv2.putText(vis_image, "STABLE - Press SPACE", (40, 95),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    status_text = "Position document in frame"
                
                cv2.putText(vis_image, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(vis_image, "Press 'q' to quit | 'e' to toggle edges", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Document Scanner', vis_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    # Capture the frame
                    if success:
                        captured_image = vis_image
                        detected_corners = corners
                        print("✓ Document captured! Choose an option...")
                    else:
                        print("✗ No document detected. Try again.")
                
                elif key == ord('e'):
                    # Toggle edge detection view
                    show_edges = not show_edges
                    print(f"Edge detection view: {'ON' if show_edges else 'OFF'}")
                        
                elif key == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _select_camera(self):
        """Detect mobile phone cameras ONLY - laptop camera completely blocked."""
        print("\n" + "="*70)
        print("  � LAPTOP CAMERA BLOCKED - PHONE CAMERA REQUIRED")
        print("="*70)
        print("\n� This scanner uses PHONE CAMERAS ONLY for professional quality!")
        print("💻 Built-in laptop cameras are DISABLED\n")
        print("✅ How to Connect Your Phone Camera:\n")
        
        print("🍎 For iPhone (macOS Ventura 13.0+):")
        print("  1. Sign into same Apple ID on iPhone and Mac")
        print("  2. Enable WiFi and Bluetooth on both")
        print("  3. Place iPhone near Mac")
        print("  4. iPhone appears as 'Continuity Camera'\n")
        
        print("🤖 For Android (or Any Phone - USB/WiFi):")
        print("  1. Install 'DroidCam' or 'EpocCam' app")
        print("  2. Install companion software on computer")
        print("  3. Connect via USB or WiFi\n")
        
        print("📸 Or Use Photo Mode (EASIEST):")
        print("  1. Take photos with phone camera")
        print("  2. Transfer to computer (AirDrop/USB)")
        print("  3. Run: python scanner_app.py photo.jpg\n")
        
        print("="*70)
        print("\n🔍 Scanning for phone cameras (skipping Camera 0)...")
        available_cameras = []
        
        # COMPLETELY SKIP INDEX 0 - Only check 1-5
        for i in range(1, 6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    available_cameras.append({
                        'index': i,
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                cap.release()
        
        if not available_cameras:
            print("\n" + "="*70)
            print("❌ ERROR: NO PHONE CAMERA DETECTED")
            print("="*70)
            print("\n⚠️  Laptop camera is PERMANENTLY DISABLED")
            print("📱 You MUST connect a phone camera to use this scanner\n")
            print("Solutions:\n")
            print("  1️⃣  Connect iPhone (Continuity Camera)")
            print("  2️⃣  Install DroidCam/EpocCam for USB connection")
            print("  3️⃣  Use Photo Mode (RECOMMENDED):")
            print("      • Take photos with phone camera")
            print("      • Run: python scanner_app.py photo.jpg\n")
            print("📖 Full setup guide: PHONE_CAMERA_SETUP.md")
            print("="*70 + "\n")
            print("\n🛑 Exiting - Please connect phone camera and try again!\n")
            sys.exit(1)
        
        # Mobile cameras found!
        if len(available_cameras) == 1:
            cam = available_cameras[0]
            print(f"\n✓ Mobile camera detected!")
            print(f"  Camera {cam['index']} - {cam['resolution']} @ {cam['fps']}fps")
            print(f"\n🎉 Using mobile camera for high-quality scanning!\n")
            return cam['index']
        
        # Multiple mobile cameras - let user choose
        print(f"\n✓ Found {len(available_cameras)} mobile cameras:\n")
        for i, cam in enumerate(available_cameras):
            camera_name = f"Mobile Camera {cam['index']}"
            print(f"  {i+1}. {camera_name} - {cam['resolution']} @ {cam['fps']}fps")
        
        while True:
            try:
                choice = input(f"\nSelect camera (1-{len(available_cameras)}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_cameras):
                    selected = available_cameras[choice_idx]
                    print(f"\n✓ Using mobile camera {selected['index']}")
                    print(f"🎉 Ready for high-quality scanning!\n")
                    return selected['index']
                else:
                    print(f"Please enter a number between 1 and {len(available_cameras)}")
            except ValueError:
                print("Please enter a valid number")
    
    def scan_from_file(self, image_path):
        """Scan a document from an image file."""
        print(f"\n=== Scanning from file: {image_path} ===")
        
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        # Try with more relaxed settings for difficult images
        original_min_area = self.detector.min_area_ratio
        
        # Detect document with standard settings
        print("Detecting document (standard settings)...")
        success, corners, vis_image = self.detector.detect(image, use_temporal_smoothing=False)
        
        if not success:
            # Try again with more relaxed settings
            print("Trying with relaxed detection settings...")
            self.detector.min_area_ratio = 0.01  # Much more sensitive
            success, corners, vis_image = self.detector.detect(image, use_temporal_smoothing=False)
            self.detector.min_area_ratio = original_min_area
        
        if not success:
            print("✗ No document detected in the image")
            print("\nTips to improve detection:")
            print("  • Ensure document has good contrast with background")
            print("  • Use a dark surface under white paper")
            print("  • Make sure entire document is visible")
            print("  • Ensure good, even lighting")
            print("  • Try taking photo from directly above")
            cv2.imshow('Detection Failed - Press any key', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        
        print("✓ Document detected successfully!")
        
        # Show detection
        cv2.imshow('Detected Document', vis_image)
        cv2.waitKey(1)
        
        # Show enhancement options
        self._show_enhancement_options(image, corners)
    
    def _show_enhancement_options(self, image, corners):
        """Show different enhancement options to user."""
        print("\nGenerating previews...")
        
        # Generate different versions
        versions = {
            'Original (Color)': self.corrector.correct(image, corners, enhance_method='none'),
            'Enhanced Color': self.corrector.correct(image, corners, enhance_method='color'),
            'Black & White (Adaptive)': self.corrector.correct(image, corners, enhance_method='adaptive'),
            'Black & White (Otsu)': self.corrector.correct(image, corners, enhance_method='otsu')
        }
        
        # Display options
        print("\n=== Enhancement Options ===")
        print("1. Original (Color)")
        print("2. Enhanced Color")
        print("3. Black & White (Adaptive Threshold)")
        print("4. Black & White (Otsu Threshold)")
        print("\nPress 1-4 to select, or 'q' to cancel")
        
        current_option = 0
        option_list = list(versions.keys())
        
        while True:
            # Show current option
            display = versions[option_list[current_option]].copy()
            
            # Add text
            cv2.putText(display, option_list[current_option], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press 1-4 to switch | SPACE to save | 'q' to cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Select Enhancement', display)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('1'):
                current_option = 0
            elif key == ord('2'):
                current_option = 1
            elif key == ord('3'):
                current_option = 2
            elif key == ord('4'):
                current_option = 3
            elif key == ord(' '):
                # Save current option
                self._save_image(versions[option_list[current_option]], option_list[current_option])
                break
            elif key == ord('q'):
                print("Cancelled")
                break
        
        cv2.destroyAllWindows()
    
    def _save_scanned_document(self, image, corners):
        """Save scanned document with default settings."""
        corrected = self.corrector.correct(image, corners, enhance_method='adaptive')
        self._save_image(corrected, 'Black & White (Adaptive)')
    
    def _save_image(self, image, description):
        """Save image to output directory."""
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scanned_doc_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        print(f"\n✓ Saved: {filepath}")
        print(f"  Enhancement: {description}")
    
    def batch_process(self, input_dir, output_dir=None):
        """Process multiple images in batch mode."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'batch')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image formats
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"\n=== Batch Processing {len(image_files)} images ===\n")
        
        success_count = 0
        for i, filename in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing {filename}...")
            
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                print(f"  ✗ Could not read image")
                continue
            
            # Detect and correct
            success, corners, _ = self.detector.detect(image)
            
            if success:
                corrected = self.corrector.correct(image, corners, enhance_method='adaptive')
                
                # Save with same name
                output_path = os.path.join(output_dir, f"scanned_{filename}")
                cv2.imwrite(output_path, corrected)
                
                print(f"  ✓ Saved to {output_path}")
                success_count += 1
            else:
                print(f"  ✗ No document detected")
        
        print(f"\n=== Batch Complete: {success_count}/{len(image_files)} successful ===")


def print_usage():
    """Print usage information."""
    print("\n" + "="*60)
    print("  AUTOMATIC DOCUMENT SCANNER & PERSPECTIVE CORRECTION")
    print("="*60)
    print("\nUsage:")
    print("  python scanner_app.py                    - Webcam mode")
    print("  python scanner_app.py <image_path>       - Scan single image")
    print("  python scanner_app.py batch <input_dir>  - Batch process directory")
    print("\nExamples:")
    print("  python scanner_app.py")
    print("  python scanner_app.py ../samples/document.jpg")
    print("  python scanner_app.py batch ../samples/")
    print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    scanner = DocumentScanner()
    
    if len(sys.argv) == 1:
        # Webcam mode
        scanner.scan_from_webcam()
    elif len(sys.argv) == 2:
        # Single file mode
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            print_usage()
            return
        scanner.scan_from_file(image_path)
    elif len(sys.argv) == 3 and sys.argv[1] == 'batch':
        # Batch mode
        input_dir = sys.argv[2]
        if not os.path.isdir(input_dir):
            print(f"Error: Directory not found: {input_dir}")
            print_usage()
            return
        scanner.batch_process(input_dir)
    else:
        print_usage()


if __name__ == "__main__":
    main()
